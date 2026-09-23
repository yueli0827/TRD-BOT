import argparse
import json
from pathlib import Path

import numpy as np

from clustering.pipeline import cluster_frames, estimate_cluster_count, load_feature_file
from postpromoting.src.bot import segment_bot


def run_video(
    basic_features,
    trd_features,
    video_id,
    cache_dir,
    output_dir,
    backend="spectral",
    seed=0,
    deepdpm_repo=None,
    deepdpm_python=None,
    deepdpm_args=None,
    deepdpm_timeout=None,
    save_transport=False,
    bot_options=None,
):
    destination = Path(output_dir)
    for name in ("basic", "trd_only", "bot_only", "trd_bot"):
        path = destination / f"{name}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                if "video_id" not in archive or str(archive["video_id"]) != str(video_id):
                    raise FileExistsError(f"Output belongs to another video; choose a separate output directory: {path}")
    basic = load_feature_file(Path(basic_features))
    trd = load_feature_file(Path(trd_features))
    if basic.shape != trd.shape:
        raise ValueError("Basic and TRD descriptors must have the same frame count and dimension.")
    count = estimate_cluster_count(
        basic,
        video_id,
        cache_dir,
        deepdpm_repo=deepdpm_repo,
        deepdpm_python=deepdpm_python,
        deepdpm_args=deepdpm_args,
        seed=seed,
        timeout=deepdpm_timeout,
    )
    basic_labels = cluster_frames(basic, count, backend=backend, seed=seed)
    trd_labels = cluster_frames(trd, count, backend=backend, seed=seed)
    options = {} if bot_options is None else dict(bot_options)
    results = {
        "basic": {"labels": basic_labels},
        "trd_only": {"labels": trd_labels},
        "bot_only": segment_bot(basic, basic_labels, **options),
        "trd_bot": segment_bot(trd, trd_labels, **options),
    }
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, result in results.items():
        payload = {key: value for key, value in result.items() if save_transport or key != "transport"}
        payload.update(video_id=str(video_id), n_clusters=count, backend=backend, seed=seed)
        path = destination / f"{name}.npz"
        np.savez_compressed(path, **payload)
        paths[name] = str(path.resolve())
    return {"video_id": str(video_id), "n_clusters": count, "outputs": paths}


def main():
    parser = argparse.ArgumentParser(description="Run matched Basic, TRD, BOT and TRD-BOT segmentation for one video.")
    parser.add_argument("--basic-features", type=Path, required=True)
    parser.add_argument("--trd-features", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("kmeans", "gmm", "spectral", "graph", "hierarchical"), default="spectral")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deepdpm-repo", type=Path)
    parser.add_argument("--deepdpm-python", type=Path)
    parser.add_argument("--deepdpm-arg", action="append", default=[])
    parser.add_argument("--deepdpm-timeout", type=float)
    parser.add_argument("--kappa", type=float, default=0.15)
    parser.add_argument("--structure-weight", type=float, default=0.65)
    parser.add_argument("--entropy-weight", type=float, default=0.03)
    parser.add_argument("--windows", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--temporal-width", type=int)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--disable-boundary-gating", action="store_true")
    parser.add_argument("--save-transport", action="store_true")
    args = parser.parse_args()
    summary = run_video(
        args.basic_features,
        args.trd_features,
        args.video_id,
        args.cache_dir,
        args.output_dir,
        backend=args.backend,
        seed=args.seed,
        deepdpm_repo=args.deepdpm_repo,
        deepdpm_python=args.deepdpm_python,
        deepdpm_args=args.deepdpm_arg,
        deepdpm_timeout=args.deepdpm_timeout,
        save_transport=args.save_transport,
        bot_options={
            "kappa": args.kappa,
            "structure_weight": args.structure_weight,
            "entropy_weight": args.entropy_weight,
            "windows": tuple(args.windows),
            "temporal_width": args.temporal_width,
            "max_iterations": args.max_iterations,
            "use_boundary_gating": not args.disable_boundary_gating,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
