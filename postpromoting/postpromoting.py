import argparse
from pathlib import Path

import numpy as np

if __package__:
    from .src.bot import segment_bot
else:
    from src.bot import segment_bot


def load_array(path, key):
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if key not in loaded.files:
                raise ValueError(f"{path} does not contain the array {key!r}")
            return loaded[key]
        finally:
            loaded.close()
    return loaded


def main():
    parser = argparse.ArgumentParser(description="Boundary-guided optimal transport for one video")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features-key", default="features")
    parser.add_argument("--labels-key", default="labels")
    parser.add_argument("--kappa", type=float, default=0.15)
    parser.add_argument("--structure-weight", type=float, default=0.65)
    parser.add_argument("--entropy-weight", type=float, default=0.03)
    parser.add_argument("--windows", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--temporal-width", type=int)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--sinkhorn-iterations", type=int, default=1000)
    parser.add_argument("--step-size", type=float, default=10.0)
    parser.add_argument("--objective-tolerance", type=float, default=1e-8)
    parser.add_argument("--marginal-tolerance", type=float, default=1e-9)
    parser.add_argument("--max-backtracking", type=int, default=30)
    parser.add_argument("--disable-boundary-gating", action="store_true")
    parser.add_argument("--save-transport", action="store_true")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must end in .npz")
    result = segment_bot(
        load_array(args.features, args.features_key),
        load_array(args.labels, args.labels_key),
        kappa=args.kappa,
        structure_weight=args.structure_weight,
        entropy_weight=args.entropy_weight,
        windows=args.windows,
        temporal_width=args.temporal_width,
        max_iterations=args.max_iterations,
        sinkhorn_iterations=args.sinkhorn_iterations,
        step_size=args.step_size,
        objective_tolerance=args.objective_tolerance,
        marginal_tolerance=args.marginal_tolerance,
        max_backtracking=args.max_backtracking,
        use_boundary_gating=not args.disable_boundary_gating,
    )
    if not args.save_transport:
        result.pop("transport")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(f"Saved {len(result['labels'])} frame labels and {len(result['segment_labels'])} segments to {args.output}")
    print(f"Solver: {result['solver_status']}; marginal error: {float(result['marginal_error']):.3e}")


if __name__ == "__main__":
    main()
