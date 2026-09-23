import argparse
from pathlib import Path
import time

import numpy as np
import torch

if __package__:
    from .src.trd import TRDModel
    from .src.video_pairs import VideoFrames, frame_tensor, validate_preprocessing
else:
    from src.trd import TRDModel
    from src.video_pairs import VideoFrames, frame_tensor, validate_preprocessing


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ordered, normalized 13-layer student frame descriptors")
    parser.add_argument("--video_path", type=Path, required=True, help="Video file or ordered frame directory")
    parser.add_argument("--ckpt_path", type=Path, required=True, help="Student checkpoint from train_trd.py")
    parser.add_argument("--model_id", help="Optional local SD2.1 model directory for the frozen VAE")
    parser.add_argument("--num_frames", type=int, help="Uniformly sample at most this many frames; default is every frame")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--crop_percent", type=float, nargs=4, metavar=("LEFT", "RIGHT", "TOP", "BOTTOM"))
    parser.add_argument("--output_base_path", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or (args.num_frames is not None and args.num_frames < 1):
        raise ValueError("batch_size and num_frames must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and args.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bfloat16; use --precision float32")
    source = VideoFrames(args.video_path)
    count = min(args.num_frames, source.frame_count) if args.num_frames else source.frame_count
    indices = np.linspace(0, source.frame_count - 1, count, dtype=np.int64)
    model, checkpoint = TRDModel.from_checkpoint(args.ckpt_path, device=device,
                                                 mixed_precision=args.precision == "bfloat16",
                                                 model_id=args.model_id,
                                                 local_files_only=args.local_files_only)
    image_size = args.image_size or checkpoint["image_size"]
    crop_percent = args.crop_percent or checkpoint["crop_percent"]
    validate_preprocessing(image_size, crop_percent)
    output_dir = args.output_base_path or Path("features_new_data") / args.ckpt_path.stem / args.video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "video_features.npy"
    temporary_path = output_dir / "video_features.tmp.npy"
    descriptors = None
    start_time = time.perf_counter()
    try:
        for start in range(0, count, args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            frames = source.read(batch_indices)
            batch = torch.stack([frame_tensor(frame, image_size, crop_percent) for frame in frames]).to(device)
            values = model.extract(batch).cpu().numpy().astype(np.float32, copy=False)
            if not np.isfinite(values).all():
                raise FloatingPointError("Feature extraction produced nonfinite values")
            if descriptors is None:
                descriptors = np.lib.format.open_memmap(temporary_path, mode="w+", dtype=np.float32,
                                                        shape=(count, values.shape[1]))
            descriptors[start:start + len(values)] = values
            if start == 0 or start + len(values) == count or (start // args.batch_size + 1) % 50 == 0:
                print(f"Extracted {start + len(values)}/{count} frames", flush=True)
        descriptors.flush()
        del descriptors
        descriptors = None
        temporary_path.replace(output_path)
        np.save(output_dir / "frame_indices.npy", indices, allow_pickle=False)
    finally:
        if descriptors is not None:
            del descriptors
    print(f"Saved {count} descriptors to {output_path} in {time.perf_counter() - start_time:.2f}s")


if __name__ == "__main__":
    main()
