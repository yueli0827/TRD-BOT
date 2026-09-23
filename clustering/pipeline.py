import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


BACKENDS = ("spectral", "kmeans", "gmm", "graph", "hierarchical")


def normalize_features(features):
    features = np.asarray(features)
    if features.ndim != 2 or min(features.shape) < 1:
        raise ValueError("Expected one video's features with shape [frames, channels].")
    if not np.issubdtype(features.dtype, np.number) or np.iscomplexobj(features):
        raise ValueError("Features must be real numbers.")
    features = np.asarray(features, dtype=np.float32, order="C")
    if not np.all(np.isfinite(features)):
        raise ValueError("Features contain NaN or infinity.")
    norms = np.sqrt(np.einsum("ij,ij->i", features, features, dtype=np.float64))[:, None]
    if np.any(norms <= 1e-12):
        raise ValueError("Features contain zero-length frame vectors.")
    if np.all(np.abs(norms - 1.0) <= 1e-6):
        return features
    return np.divide(features, norms, out=np.empty_like(features), casting="unsafe")


def load_feature_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Expected one feature file per video: {path}")
    if path.suffix.lower() == ".npy":
        features = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "features" not in archive:
                raise ValueError("A feature archive must contain a 'features' array.")
            features = archive["features"]
    elif path.suffix.lower() in (".pt", ".pth"):
        import torch

        features = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(features, dict):
            if "features" not in features:
                raise ValueError("A feature checkpoint must contain a 'features' tensor.")
            features = features["features"]
        if not isinstance(features, torch.Tensor):
            raise ValueError("Expected a tensor containing one video's features.")
        features = features.detach().cpu().numpy()
    elif path.suffix.lower() in (".csv", ".txt"):
        delimiter = "," if path.suffix.lower() == ".csv" else None
        features = np.loadtxt(path, delimiter=delimiter, ndmin=2)
    else:
        raise ValueError(f"Unsupported feature format: {path.suffix}")
    return normalize_features(features)


def feature_fingerprint(features):
    features = np.asarray(features, dtype="<f4", order="C")
    digest = hashlib.sha256()
    digest.update(np.asarray(features.shape, dtype="<i8").tobytes())
    digest.update(memoryview(features).cast("B"))
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _checked_count(value, frames):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("Cluster count must be an integer.")
    if not 1 <= value <= frames:
        raise ValueError(f"Cluster count must be between 1 and {frames}.")
    return int(value)


def _load_cached_count(path, video_id, fingerprint, shape):
    with Path(path).open(encoding="utf-8") as stream:
        record = json.load(stream)
    expected = {"version": 1, "video_id": video_id, "basic_feature_sha256": fingerprint, "shape": list(shape)}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Cluster-count cache does not match this video's Basic features: {path}")
    if record.get("estimator") != "DeepDPM":
        raise ValueError(f"Cluster count was not produced by the DeepDPM adapter: {path}")
    return _checked_count(record.get("n_clusters"), shape[0])


def _deepdpm_arguments(arguments):
    arguments = list(arguments or ())
    controlled = {
        "--dir", "--dataset", "--offline", "--seed", "--transform_input_data",
        "--use_labels_for_eval", "--save_checkpoints", "--exp_name", "--log_emb",
    }
    for argument in arguments:
        if not isinstance(argument, str):
            raise TypeError("DeepDPM arguments must be a sequence of strings.")
        flag = argument.split("=", 1)[0]
        if flag.startswith("--") and any(option.startswith(flag) for option in controlled):
            raise ValueError(f"The per-video adapter controls this DeepDPM option: {argument}")
    return arguments


def _python_executable(value):
    executable = str(value or sys.executable)
    path = Path(executable).expanduser()
    if path.is_file():
        return str(path.resolve())
    if path.is_absolute() or path.parent != Path("."):
        raise FileNotFoundError(f"DeepDPM Python executable was not found: {path}")
    return executable


def estimate_cluster_count(
    basic_features,
    video_id,
    cache_dir,
    *,
    deepdpm_repo=None,
    deepdpm_python=None,
    deepdpm_args=None,
    seed=0,
    timeout=None,
):
    basic_features = normalize_features(basic_features)
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("A nonempty video_id is required for the shared count cache.")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = feature_fingerprint(basic_features)
    key = hashlib.sha256((video_id + "\0" + fingerprint).encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists():
        return _load_cached_count(cache_path, video_id, fingerprint, basic_features.shape)
    if deepdpm_repo is None:
        raise FileNotFoundError("No cached DeepDPM count exists. Supply the official DeepDPM repository and its Python environment.")
    repository = Path(deepdpm_repo).resolve()
    if not (repository / "DeepDPM.py").is_file() or not (repository / "src" / "datasets.py").is_file():
        raise FileNotFoundError(f"Expected an official DeepDPM checkout: {repository}")
    arguments = _deepdpm_arguments(deepdpm_args)
    python_executable = _python_executable(deepdpm_python)
    lock_path = cache_path.with_suffix(".lock")
    try:
        lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"DeepDPM estimation is already running for this video; retry after completion: {lock_path}") from error
    try:
        with os.fdopen(lock, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        if cache_path.exists():
            return _load_cached_count(cache_path, video_id, fingerprint, basic_features.shape)
        run_directory = cache_dir / f"{key}.deepdpm"
        run_directory.mkdir(parents=True, exist_ok=True)
        input_path = run_directory / "basic_features.npy"
        output_path = run_directory / "deepdpm_assignments.npy"
        log_path = run_directory / "deepdpm.log"
        output_path.unlink(missing_ok=True)
        np.save(input_path, basic_features, allow_pickle=False)
        command = [
            python_executable,
            str(Path(__file__).with_name("deepdpm_runner.py").resolve()),
            "--repository", str(repository),
            "--features", str(input_path.resolve()),
            "--output", str(output_path.resolve()),
            "--seed", str(int(seed)),
            "--", *arguments,
        ]
        with log_path.open("w", encoding="utf-8") as stream:
            try:
                result = subprocess.run(command, cwd=run_directory, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            except (OSError, subprocess.TimeoutExpired) as error:
                raise RuntimeError(f"DeepDPM could not complete. See {log_path}") from error
        if result.returncode != 0 or not output_path.exists():
            raise RuntimeError(f"Official DeepDPM failed with exit code {result.returncode}. See {log_path}")
        labels = np.load(output_path, allow_pickle=False)
        if labels.shape != (len(basic_features),) or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise ValueError("DeepDPM must return one nonnegative integer assignment per frame.")
        count = _checked_count(int(np.unique(labels).size), len(basic_features))
        _write_json(cache_path, {
            "version": 1,
            "video_id": video_id,
            "basic_feature_sha256": fingerprint,
            "shape": list(basic_features.shape),
            "estimator": "DeepDPM",
            "n_clusters": count,
            "seed": int(seed),
            "repository": str(repository),
            "entrypoint_sha256": hashlib.sha256((repository / "DeepDPM.py").read_bytes()).hexdigest(),
            "arguments": arguments,
            "python": python_executable,
            "assignments_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "created_at": time.time(),
        })
        return count
    finally:
        lock_path.unlink(missing_ok=True)


def build_knn_affinity(features, n_neighbors=20):
    count = len(features)
    if count < 2:
        return sparse.csr_matrix((count, count), dtype=np.float64)
    neighbors = min(int(n_neighbors), count - 1)
    if neighbors < 1:
        raise ValueError("n_neighbors must be positive.")
    model = NearestNeighbors(n_neighbors=neighbors + 1, metric="euclidean")
    distances, indices = model.fit(features).kneighbors(features)
    rows = []
    columns = []
    values = []
    scales = np.maximum(distances[:, -1], 1e-6)
    for frame in range(count):
        keep = indices[frame] != frame
        adjacent = indices[frame][keep][:neighbors]
        distance = distances[frame][keep][:neighbors]
        weights = np.exp(-distance ** 2 / (scales[frame] * scales[adjacent] + 1e-12))
        rows.extend([frame] * len(adjacent))
        columns.extend(adjacent.tolist())
        values.extend(weights.tolist())
    affinity = sparse.csr_matrix((values, (rows, columns)), shape=(count, count))
    return affinity.maximum(affinity.T)


def cluster_frames(features, n_clusters, backend="spectral", seed=0):
    features = normalize_features(features)
    n_clusters = _checked_count(n_clusters, len(features))
    if backend not in BACKENDS:
        raise ValueError(f"Unknown clustering backend: {backend}")
    if n_clusters == 1:
        return np.zeros(len(features), dtype=np.int64)
    if n_clusters == len(features):
        return np.arange(len(features), dtype=np.int64)
    if backend == "kmeans":
        estimator = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20, max_iter=300)
        labels = estimator.fit_predict(features)
    elif backend == "gmm":
        estimator = GaussianMixture(n_components=n_clusters, covariance_type="diag", reg_covar=1e-6, max_iter=200, n_init=2, random_state=seed)
        labels = estimator.fit_predict(features)
    elif backend == "spectral":
        affinity = build_knn_affinity(features)
        estimator = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", assign_labels="kmeans", n_init=20, random_state=seed)
        labels = estimator.fit_predict(affinity)
    elif backend == "hierarchical":
        labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(features)
    else:
        if __package__ in (None, ""):
            from graph_backend import cluster_graph
        else:
            from .graph_backend import cluster_graph

        labels = cluster_graph(features, n_clusters=n_clusters, seed=seed)
    return np.asarray(labels, dtype=np.int64)


def cluster_video(
    feature_path,
    basic_feature_path,
    output_path,
    cache_dir,
    *,
    video_id=None,
    backend="spectral",
    seed=0,
    deepdpm_repo=None,
    deepdpm_python=None,
    deepdpm_args=None,
    timeout=None,
):
    feature_path = Path(feature_path)
    basic_feature_path = Path(basic_feature_path)
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".npy":
        raise ValueError("Label output must use the .npy extension.")
    features = load_feature_file(feature_path)
    basic_features = load_feature_file(basic_feature_path)
    if features.shape != basic_features.shape:
        raise ValueError("Basic and target features must have matching frame and channel counts.")
    video_id = video_id or str(basic_feature_path.resolve())
    count = estimate_cluster_count(
        basic_features, video_id, cache_dir, deepdpm_repo=deepdpm_repo,
        deepdpm_python=deepdpm_python, deepdpm_args=deepdpm_args, seed=seed, timeout=timeout,
    )
    labels = cluster_frames(features, count, backend=backend, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, labels, allow_pickle=False)
    _write_json(output_path.with_suffix(".json"), {
        "video_id": video_id,
        "backend": backend,
        "seed": int(seed),
        "n_frames": len(features),
        "n_clusters_requested": count,
        "n_clusters_observed": int(np.unique(labels).size),
        "features": str(feature_path.resolve()),
        "basic_features": str(basic_feature_path.resolve()),
    })
    return labels


def main(default_backend=None, argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", "--input_path", dest="features", required=True, type=Path)
    parser.add_argument("--basic-features", required=True, type=Path)
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output", type=Path)
    output_group.add_argument("--out_dir", type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--video-id")
    parser.add_argument("--backend", choices=BACKENDS, default=default_backend or "spectral")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deepdpm-repo", type=Path)
    parser.add_argument("--deepdpm-python")
    parser.add_argument("--deepdpm-timeout", type=float)
    parser.add_argument("--deepdpm-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args(argv)
    labels = cluster_video(
        args.features, args.basic_features, args.output or args.out_dir / "labels.npy", args.cache_dir,
        video_id=args.video_id, backend=args.backend, seed=args.seed,
        deepdpm_repo=args.deepdpm_repo, deepdpm_python=args.deepdpm_python,
        deepdpm_args=args.deepdpm_args, timeout=args.deepdpm_timeout,
    )
    print(json.dumps({"frames": len(labels), "observed_clusters": int(np.unique(labels).size)}))


if __name__ == "__main__":
    main()
