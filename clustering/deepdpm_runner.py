import argparse
import importlib
import os
from pathlib import Path
import random
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    protected = {"--dir", "--dataset", "--offline", "--seed", "--transform_input_data", "--use_labels_for_eval", "--save_checkpoints", "--exp_name", "--log_emb"}
    if any(value.startswith("--") and any(option.startswith(value.split("=", 1)[0]) for option in protected) for value in extra):
        raise ValueError("DeepDPM data, normalization, logging, and evaluation options are controlled by this adapter.")
    repository = args.repository.resolve()
    features_path = args.features.resolve()
    output_path = args.output.resolve()
    if not (repository / "DeepDPM.py").is_file():
        raise FileNotFoundError(repository / "DeepDPM.py")
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    features = np.load(features_path, allow_pickle=False)
    if features.ndim != 2 or min(features.shape) < 1 or not np.all(np.isfinite(features)):
        raise ValueError("Expected finite [frames, channels] Basic features.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    torch.save(tensor, output_path.parent / "train_data.pt")
    torch.save(tensor, output_path.parent / "test_data.pt")
    os.chdir(output_path.parent)
    sys.path.insert(0, str(repository))
    entrypoint = importlib.import_module("DeepDPM")
    sys.argv = [
        str(repository / "DeepDPM.py"), "--dataset", "custom", "--dir", str(output_path.parent),
        "--offline", "--seed", str(args.seed), "--transform_input_data", "normalize", "--log_emb", "never", *extra,
    ]
    labels = np.asarray(entrypoint.train_cluster_net())
    if labels.shape != (len(features),) or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("Official DeepDPM returned invalid assignments.")
    np.save(output_path, labels.astype(np.int64), allow_pickle=False)


if __name__ == "__main__":
    main()
