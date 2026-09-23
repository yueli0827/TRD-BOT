import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

if __package__:
    from .src.trd import TRDModel
    from .src.video_pairs import AdjacentFramePairs, discover_videos
else:
    from src.trd import TRDModel
    from src.video_pairs import AdjacentFramePairs, discover_videos


def parse_args():
    parser = argparse.ArgumentParser(description="Train one TRD or Basic student on a dataset's unlabeled videos")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--dataset_dir", type=Path)
    sources.add_argument("--video_paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset_name")
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--beta", type=float, default=0.1, help="Use 0 for the matched Basic student")
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--training_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--teacher_timesteps", type=int, default=5)
    parser.add_argument("--student_timestep", type=float, default=261.0)
    parser.add_argument("--image_size", type=int, nargs=2, default=(384, 384), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--crop_percent", type=float, nargs=4, default=(0, 0, 0, 0))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint_every", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.training_steps < 1 or args.gradient_accumulation_steps < 1 or args.learning_rate <= 0:
        raise ValueError("Training steps, accumulation steps, and learning rate must be positive")
    if args.beta < 0 or args.warmup_steps < 0 or args.num_workers < 0 or args.checkpoint_every < 0:
        raise ValueError("beta, warmup steps, worker count, and checkpoint interval must be nonnegative")
    if args.log_every < 1 or not 1 <= args.teacher_timesteps <= 1000:
        raise ValueError("Invalid log interval or teacher timestep count")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and args.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bfloat16; use --precision float32")
    paths = discover_videos(args.dataset_dir) if args.dataset_dir else args.video_paths
    dataset = AdjacentFramePairs(paths, args.image_size, args.crop_percent)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = RandomSampler(dataset, replacement=True,
                            num_samples=args.training_steps * args.gradient_accumulation_steps,
                            generator=generator)
    loader = DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    model = TRDModel.from_pretrained(args.model_id, device=device, beta=args.beta,
                                     timestep=args.student_timestep, teacher_timesteps=args.teacher_timesteps,
                                     mixed_precision=args.precision == "bfloat16",
                                     local_files_only=args.local_files_only).train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    iterator = iter(loader)
    dataset_name = args.dataset_name or (args.dataset_dir.name if args.dataset_dir else None)
    for step in range(args.training_steps):
        multiplier = min(1.0, (step + 1) / args.warmup_steps) if args.warmup_steps else 1.0
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * multiplier
        optimizer.zero_grad(set_to_none=True)
        totals = {"feature_alignment": 0.0, "temporal_relation": 0.0, "loss": 0.0}
        for _ in range(args.gradient_accumulation_steps):
            frames = next(iterator).to(device, non_blocking=True)
            values = model.backward_pair(frames, loss_scale=1 / args.gradient_accumulation_steps)
            for key in totals:
                totals[key] += values[key] / args.gradient_accumulation_steps
        gradients_finite = torch.stack([torch.isfinite(parameter.grad).all()
                                        for parameter in parameters if parameter.grad is not None]).all()
        if not bool(gradients_finite):
            raise FloatingPointError(f"Nonfinite gradient at training step {step + 1}")
        optimizer.step()
        with torch.no_grad():
            model.timestep.clamp_(0, model.scheduler.config.num_train_timesteps - 1)
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.training_steps:
            print(f"step={step + 1} loss={totals['loss']:.6f} "
                  f"fa={totals['feature_alignment']:.6f} trd={totals['temporal_relation']:.6f} "
                  f"t_student={float(model.timestep.detach()):.6f}", flush=True)
        if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0:
            checkpoint = args.output.with_name(f"{args.output.stem}_step_{step + 1}{args.output.suffix}")
            model.save_student(checkpoint, args.image_size, args.crop_percent, step + 1, dataset_name)
    model.save_student(args.output, args.image_size, args.crop_percent, args.training_steps, dataset_name)
    print(f"Saved student checkpoint: {args.output}", flush=True)


if __name__ == "__main__":
    main()
