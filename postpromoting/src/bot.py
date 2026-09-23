import math

import numpy as np
from scipy import sparse
from scipy.special import logsumexp, xlogy


def contiguous_segments(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("labels must be a nonempty one-dimensional array")
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[starts[1:], len(labels)]
    return starts, ends, labels[starts]


def normalize_rows(features, delta=1e-12):
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), delta)


def boundary_reliability(features, windows=(5, 10, 20), kappa=0.15, delta=1e-12):
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    windows = tuple(windows)
    if not windows or any(int(value) != value or value <= 0 for value in windows):
        raise ValueError("windows must contain positive integers")
    features = np.asarray(features)
    count = len(features)
    if count == 1:
        return np.empty(0, dtype=np.float64)
    squared_norms = np.einsum("ij,ij->i", features, features, dtype=np.float64)
    norms = np.maximum(np.sqrt(squared_norms), delta)
    norm_prefix = np.r_[0.0, np.cumsum(squared_norms / np.square(norms))]
    boundaries = np.arange(1, count)
    sizes_by_scale = [np.minimum(np.minimum(int(window), boundaries), count - boundaries) for window in windows]
    left_norms = np.zeros((len(windows), count - 1))
    right_norms = np.zeros_like(left_norms)
    differences = np.zeros_like(left_norms)
    for start in range(0, features.shape[1], 256):
        normalized = features[:, start:start + 256] / norms[:, None]
        prefix = np.empty((count + 1, normalized.shape[1]), dtype=np.float64)
        prefix[0] = 0.0
        np.cumsum(normalized, axis=0, out=prefix[1:])
        for scale, sizes in enumerate(sizes_by_scale):
            left_mean = (prefix[boundaries] - prefix[boundaries - sizes]) / sizes[:, None]
            right_mean = (prefix[boundaries + sizes] - prefix[boundaries]) / sizes[:, None]
            left_norms[scale] += np.einsum("ij,ij->i", left_mean, left_mean)
            right_norms[scale] += np.einsum("ij,ij->i", right_mean, right_mean)
            difference = left_mean - right_mean
            differences[scale] += np.einsum("ij,ij->i", difference, difference)
    scores = []
    for scale, sizes in enumerate(sizes_by_scale):
        left_var = (norm_prefix[boundaries] - norm_prefix[boundaries - sizes]) / sizes
        left_var -= left_norms[scale]
        right_var = (norm_prefix[boundaries + sizes] - norm_prefix[boundaries]) / sizes
        right_var -= right_norms[scale]
        change = differences[scale] - 0.5 * (np.maximum(left_var, 0.0) + np.maximum(right_var, 0.0))
        scores.append(np.maximum(change, 0.0))
    median = np.median(np.stack(scores), axis=0)
    return median / (median + kappa)


def temporal_affinity(boundary_scores, width):
    boundary_scores = np.asarray(boundary_scores, dtype=np.float64)
    if boundary_scores.ndim != 1 or not np.all(np.isfinite(boundary_scores)):
        raise ValueError("boundary scores must be a finite one-dimensional array")
    if np.any(boundary_scores < 0) or np.any(boundary_scores > 1):
        raise ValueError("boundary scores must lie in [0, 1]")
    if int(width) != width or width < 1:
        raise ValueError("temporal width must be a positive integer")
    count = len(boundary_scores) + 1
    diagonals = [np.full(count, 0.5)]
    offsets = [0]
    maxima = boundary_scores.copy()
    for distance in range(1, min(int(width), count)):
        if distance > 1:
            maxima = np.maximum(maxima[:-1], boundary_scores[distance - 1:])
        weights = (1.0 - maxima) * (0.5 - distance / (2.0 * width))
        diagonals.extend((weights, weights))
        offsets.extend((distance, -distance))
    return sparse.diags(diagonals, offsets, shape=(count, count), format="csr")


class BOTProblem:
    def __init__(
        self,
        features,
        labels,
        kappa=0.15,
        structure_weight=0.65,
        entropy_weight=0.03,
        windows=(5, 10, 20),
        temporal_width=None,
        use_boundary_gating=True,
        delta=1e-12,
    ):
        features = np.asarray(features)
        labels = np.asarray(labels)
        if features.dtype.kind not in "biuf" or features.ndim != 2 or min(features.shape) < 1 or not np.all(np.isfinite(features)):
            raise ValueError("features must be a nonempty finite frame-by-feature matrix")
        if labels.ndim != 1 or len(labels) != len(features):
            raise ValueError("labels must have one entry per frame")
        if labels.dtype.kind not in "biufUS":
            raise ValueError("labels must be numeric or string values without Python objects")
        if labels.dtype.kind in "f" and not np.all(np.isfinite(labels)):
            raise ValueError("labels must be finite")
        if not np.isfinite(structure_weight) or not 0 <= structure_weight <= 1:
            raise ValueError("structure_weight must lie in [0, 1]")
        if not np.isfinite(entropy_weight) or entropy_weight <= 0:
            raise ValueError("entropy_weight must be finite and positive")
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("delta must be finite and positive")
        self.starts, self.ends, self.segment_labels = contiguous_segments(labels)
        self.initial_labels = labels.copy()
        lengths = self.ends - self.starts
        centers = np.add.reduceat(features, self.starts, axis=0, dtype=np.float64) / lengths[:, None]
        centers = normalize_rows(centers, delta)
        feature_norms = np.maximum(np.sqrt(np.einsum("ij,ij->i", features, features, dtype=np.float64)), delta)
        similarity = np.empty((len(features), len(centers)), dtype=np.float64)
        chunk_size = max(1, 8388608 // features.shape[1])
        for start in range(0, len(features), chunk_size):
            end = start + chunk_size
            similarity[start:end] = (features[start:end] @ centers.T) / feature_norms[start:end, None]
        self.visual_cost = 1.0 - np.clip(similarity, -1.0, 1.0)
        self.frame_marginal = np.full(len(features), 1.0 / len(features))
        self.segment_marginal = lengths.astype(np.float64) / len(features)
        self.boundary_scores = boundary_reliability(features, windows, kappa, delta)
        self.temporal_width = max(1, math.ceil(0.02 * len(features))) if temporal_width is None else temporal_width
        self.affinity = temporal_affinity(
            self.boundary_scores if use_boundary_gating else np.zeros_like(self.boundary_scores),
            self.temporal_width,
        )
        self.affinity_plus_square = self.affinity + self.affinity.multiply(self.affinity)
        _, self.label_indices = np.unique(self.segment_labels, return_inverse=True)
        self.label_membership = sparse.csr_matrix(
            (
                np.ones(len(self.segment_labels)),
                (np.arange(len(self.segment_labels)), self.label_indices),
            ),
            shape=(len(self.segment_labels), int(self.label_indices.max()) + 1),
        )
        self.structure_weight = structure_weight
        self.entropy_weight = entropy_weight
        self.use_boundary_gating = bool(use_boundary_gating)

    def structural_gradient(self, transport):
        row_mass = transport.sum(axis=1)
        label_mass = self.label_membership.T.dot(transport.T).T
        shared = 0.25 * row_mass.sum() + self.affinity_plus_square.dot(row_mass)
        return shared[:, None] - 2.0 * self.affinity.dot(label_mass)[:, self.label_indices]

    def objective(self, transport):
        structural = 0.5 * np.sum(transport * self.structural_gradient(transport))
        visual = np.sum(self.visual_cost * transport)
        entropy_penalty = np.sum(xlogy(transport, transport))
        return float(
            self.structure_weight * structural
            + (1.0 - self.structure_weight) * visual
            + self.entropy_weight * entropy_penalty
        )

    def gradient(self, transport, log_transport=None):
        if log_transport is None:
            if np.any(transport <= 0):
                raise ValueError("the entropy gradient requires strictly positive transport")
            log_transport = np.log(transport)
        return (
            self.structure_weight * self.structural_gradient(transport)
            + (1.0 - self.structure_weight) * self.visual_cost
            + self.entropy_weight * (log_transport + 1.0)
        )


def marginal_error(transport, frame_marginal, segment_marginal):
    return float(max(
        np.max(np.abs(transport.sum(axis=1) - frame_marginal)),
        np.max(np.abs(transport.sum(axis=0) - segment_marginal)),
    ))


def log_sinkhorn_projection(
    log_kernel,
    frame_marginal,
    segment_marginal,
    tolerance=1e-9,
    max_iterations=1000,
    initial_log_v=None,
    return_scaling=False,
):
    log_kernel = log_kernel - np.max(log_kernel, axis=1, keepdims=True)
    log_a = np.log(frame_marginal)
    log_b = np.log(segment_marginal)
    log_v = np.zeros_like(log_b) if initial_log_v is None else np.asarray(initial_log_v).copy()
    for iteration in range(max_iterations):
        log_u = log_a - logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_b - logsumexp(log_kernel + log_u[:, None], axis=0)
        if iteration % 5 == 0 or iteration == max_iterations - 1:
            log_transport = log_kernel + log_u[:, None] + log_v[None, :]
            transport = np.exp(log_transport)
            residual = marginal_error(transport, frame_marginal, segment_marginal)
            if residual <= tolerance:
                break
    result = (transport, log_transport, residual)
    return (*result, log_v - log_v.mean()) if return_scaling else result


def segment_bot(
    features,
    labels,
    kappa=0.15,
    structure_weight=0.65,
    entropy_weight=0.03,
    windows=(5, 10, 20),
    temporal_width=None,
    use_boundary_gating=True,
    max_iterations=300,
    sinkhorn_iterations=1000,
    step_size=10.0,
    objective_tolerance=1e-8,
    marginal_tolerance=1e-9,
    max_backtracking=30,
):
    for name, value in (
        ("max_iterations", max_iterations),
        ("sinkhorn_iterations", sinkhorn_iterations),
        ("max_backtracking", max_backtracking),
    ):
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
        ("step_size", step_size),
        ("objective_tolerance", objective_tolerance),
        ("marginal_tolerance", marginal_tolerance),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    problem = BOTProblem(
        features,
        labels,
        kappa=kappa,
        structure_weight=structure_weight,
        entropy_weight=entropy_weight,
        windows=windows,
        temporal_width=temporal_width,
        use_boundary_gating=use_boundary_gating,
    )
    transport = problem.frame_marginal[:, None] * problem.segment_marginal[None, :]
    log_transport = np.log(transport)
    objective = problem.objective(transport)
    history = [objective]
    accepted_steps = []
    status = "iteration_limit"
    step = float(step_size)
    previous_scaling = np.zeros_like(problem.segment_marginal)
    previous_step = step
    for _ in range(int(max_iterations)):
        gradient = problem.gradient(transport, log_transport)
        accepted = False
        for _ in range(int(max_backtracking)):
            candidate, log_candidate, residual, candidate_scaling = log_sinkhorn_projection(
                log_transport - step * gradient,
                problem.frame_marginal,
                problem.segment_marginal,
                tolerance=marginal_tolerance,
                max_iterations=int(sinkhorn_iterations),
                initial_log_v=previous_scaling * (step / previous_step),
                return_scaling=True,
            )
            candidate_objective = problem.objective(candidate)
            if residual <= marginal_tolerance and np.isfinite(candidate_objective) and candidate_objective <= objective:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            status = "backtracking_stopped"
            break
        relative_change = abs(candidate_objective - objective) / max(abs(objective), 1e-12)
        transport, log_transport, objective = candidate, log_candidate, candidate_objective
        previous_scaling, previous_step = candidate_scaling, step
        history.append(objective)
        accepted_steps.append(step)
        if relative_change <= objective_tolerance:
            status = "converged"
            break
        step = min(step * 1.25, step_size)
    final_labels = problem.segment_labels[np.argmax(transport, axis=1)]
    starts, ends, segment_labels = contiguous_segments(final_labels)
    return {
        "labels": final_labels,
        "segment_starts": starts,
        "segment_ends": ends,
        "segment_labels": segment_labels,
        "initial_labels": problem.initial_labels,
        "initial_segment_starts": problem.starts,
        "initial_segment_ends": problem.ends,
        "initial_segment_labels": problem.segment_labels,
        "boundary_scores": problem.boundary_scores,
        "boundary_gating": np.array(problem.use_boundary_gating),
        "temporal_width": np.array(problem.temporal_width),
        "transport": transport,
        "frame_marginal": problem.frame_marginal,
        "segment_marginal": problem.segment_marginal,
        "objective_history": np.asarray(history),
        "accepted_step_sizes": np.asarray(accepted_steps),
        "marginal_error": np.array(marginal_error(transport, problem.frame_marginal, problem.segment_marginal)),
        "iterations": np.array(len(accepted_steps)),
        "solver_status": np.array(status),
    }
