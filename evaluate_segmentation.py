import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, silhouette_score


def _labels(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional label sequence.")
    if array.dtype.kind not in "biufUS":
        raise ValueError(f"{name} must contain numeric or string labels.")
    if array.dtype.kind in "f" and not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite labels.")
    return array


def _segments(labels):
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[starts[1:], labels.size]
    return starts, ends, labels[starts]


def _boundary_matches(predicted, target, tolerance):
    first = 0
    second = 0
    matches = 0
    while first < len(predicted) and second < len(target):
        if predicted[first] < target[second] - tolerance:
            first += 1
        elif target[second] < predicted[first] - tolerance:
            second += 1
        else:
            matches += 1
            first += 1
            second += 1
    return matches


def evaluate_segmentation(predicted, target, features=None, iou_threshold=0.5, boundary_tolerance=5, silhouette_samples=None, seed=0):
    prediction = _labels(predicted, "Predicted labels")
    truth = _labels(target, "Ground-truth labels")
    if prediction.shape != truth.shape:
        raise ValueError("Predicted and ground-truth labels must have the same length.")
    if not np.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must lie in (0, 1].")
    if isinstance(boundary_tolerance, bool) or not isinstance(boundary_tolerance, (int, np.integer)) or boundary_tolerance < 0:
        raise ValueError("Boundary tolerance must be a nonnegative integer.")
    pred_values, pred_indices = np.unique(prediction, return_inverse=True)
    true_values, true_indices = np.unique(truth, return_inverse=True)
    counts = np.zeros((len(pred_values), len(true_values)), dtype=np.int64)
    np.add.at(counts, (pred_indices, true_indices), 1)
    row, column = linear_sum_assignment(-counts)
    mapping = np.full(len(pred_values), -1, dtype=np.int64)
    mapping[row] = column
    mapped = mapping[pred_indices]
    pred_start, pred_end, pred_label = _segments(pred_indices)
    true_start, true_end, true_label = _segments(true_indices)
    matched = np.zeros(len(true_start), dtype=bool)
    true_positives = 0
    for start, end, label in zip(pred_start, pred_end, pred_label):
        intersection = np.maximum(0, np.minimum(end, true_end) - np.maximum(start, true_start))
        union = end - start + true_end - true_start - intersection
        overlap = intersection / union
        eligible = (true_label == mapping[label]) & ~matched & (overlap >= iou_threshold)
        if np.any(eligible):
            index = int(np.argmax(np.where(eligible, overlap, -1.0)))
            matched[index] = True
            true_positives += 1
    pred_boundaries = pred_start[1:]
    true_boundaries = true_start[1:]
    boundary_tp = _boundary_matches(pred_boundaries, true_boundaries, boundary_tolerance)
    boundary_count = len(pred_boundaries) + len(true_boundaries)
    result = {
        "sa": float(np.mean(mapped == true_indices)),
        "segmental_f1": float(2 * true_positives / (len(pred_start) + len(true_start))),
        "boundary_f1": float(2 * boundary_tp / boundary_count) if boundary_count else 1.0,
        "ari": float(adjusted_rand_score(true_indices, pred_indices)),
        "iou_threshold": float(iou_threshold),
        "boundary_tolerance": int(boundary_tolerance),
        "n_frames": int(prediction.size),
        "predicted_segments": int(len(pred_start)),
        "true_segments": int(len(true_start)),
        "segment_tp": true_positives,
        "boundary_tp": boundary_tp,
    }
    if features is not None:
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != prediction.size or array.shape[1] == 0 or not np.isfinite(array).all():
            raise ValueError("Features must be a finite frame-by-channel array aligned with the labels.")
        if silhouette_samples is not None and (isinstance(silhouette_samples, bool) or not isinstance(silhouette_samples, (int, np.integer)) or silhouette_samples < 2):
            raise ValueError("Silhouette sample count must be an integer of at least two.")
        indices = np.arange(len(array))
        if silhouette_samples is not None and silhouette_samples < len(array):
            indices = np.random.default_rng(seed).choice(len(array), silhouette_samples, replace=False)
        distinct = len(np.unique(pred_indices[indices]))
        result["silhouette"] = (
            float(silhouette_score(array[indices], pred_indices[indices], metric="euclidean"))
            if 1 < distinct < len(indices) else None
        )
        result["silhouette_frames"] = int(len(indices))
    return result


def _load_array(path, key):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive:
                raise ValueError(f"Missing array {key!r} in {path}.")
            return archive[key]
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    raise ValueError("Array files must use .npy or .npz format.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a completed segmentation using ground truth only for scoring.")
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--predicted-key", default="labels")
    parser.add_argument("--ground-truth-key", default="labels")
    parser.add_argument("--features-key", default="features")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--boundary-tolerance", type=int, default=5)
    parser.add_argument("--silhouette-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_segmentation(
        _load_array(args.predicted, args.predicted_key),
        _load_array(args.ground_truth, args.ground_truth_key),
        features=_load_array(args.features, args.features_key) if args.features else None,
        iou_threshold=args.iou_threshold,
        boundary_tolerance=args.boundary_tolerance,
        silhouette_samples=args.silhouette_samples,
        seed=args.seed,
    )
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
