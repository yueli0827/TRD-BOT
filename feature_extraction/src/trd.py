import copy
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .layers import FeedForwardBlock, FourierFeatures, MappingNetwork
from .min_sd21 import SD21UNetModel


FEATURE_DIMS = {
    "mid": 1280,
    **{f"us{i}": 1280 for i in range(1, 7)},
    **{f"us{i}": 640 for i in range(7, 10)},
    **{f"us{i}": 320 for i in range(10, 13)},
}


class SD21Features(SD21UNetModel):
    def forward(self, sample, timesteps, encoder_hidden_states):
        timesteps = timesteps.expand(sample.shape[0])
        embedding = self.time_embedding(self.time_proj(timesteps).to(sample.dtype))
        sample = self.conv_in(sample)
        residuals = (sample,)
        for block in self.down_blocks:
            if hasattr(block, "attentions"):
                sample, outputs = block(sample, embedding, encoder_hidden_states)
            else:
                sample, outputs = block(sample, embedding)
            residuals += tuple(outputs)
        sample = self.mid_block(sample, embedding, encoder_hidden_states)
        features = {"mid": sample}
        index = 1
        for block in self.up_blocks:
            count = len(block.resnets)
            skips = residuals[-count:]
            residuals = residuals[:-count]
            if hasattr(block, "attentions"):
                sample, outputs = block(sample, skips, embedding, encoder_hidden_states)
            else:
                sample, outputs = block(sample, skips, embedding)
            for output in outputs:
                features[f"us{index}"] = output
                index += 1
        return features


class ProjectionHead(nn.Module):
    def __init__(self, channels, condition_dim=256, depth=3):
        super().__init__()
        self.blocks = nn.ModuleList(
            FeedForwardBlock(channels, channels, condition_dim) for _ in range(depth)
        )

    def forward(self, features, condition):
        batch, channels, height, width = features.shape
        value = features.flatten(2).transpose(1, 2)
        for block in self.blocks:
            value = block(value, cond_norm=condition)
        return value.transpose(1, 2).reshape(batch, channels, height, width)


def spatial_cosine(left, right):
    if left.shape != right.shape or left.ndim != 4:
        raise ValueError("Spatial feature maps must have matching BCHW shapes")
    return F.cosine_similarity(left.float(), right.float(), dim=1, eps=1e-8).mean((1, 2))


def frame_descriptors(features, layer_names=tuple(FEATURE_DIMS)):
    vectors = [F.normalize(features[name].float().mean((2, 3)), dim=1) for name in layer_names]
    return F.normalize(torch.cat(vectors, dim=1), dim=1)


def sample_stratified_timesteps(total_steps, bins, device, generator=None):
    if bins < 1 or total_steps < bins:
        raise ValueError("The number of time intervals must lie between 1 and total_steps")
    return torch.stack([
        torch.randint(i * total_steps // bins, (i + 1) * total_steps // bins,
                      (), device=device, generator=generator)
        for i in range(bins)
    ])


def shared_pair_noise(latents, generator=None):
    if latents.shape[0] != 2:
        raise ValueError("A microbatch must contain one adjacent frame pair")
    noise = torch.randn(latents[:1].shape, device=latents.device,
                        dtype=latents.dtype, generator=generator)
    return noise.expand_as(latents)


class TRDModel(nn.Module):
    def __init__(self, student, vae, empty_prompt, teacher=None, scheduler=None,
                 feature_dims=None, beta=0.1, timestep=261.0, teacher_timesteps=5,
                 model_id="stabilityai/stable-diffusion-2-1", mixed_precision=False):
        super().__init__()
        if beta < 0:
            raise ValueError("beta must be nonnegative")
        self.student = student
        self.vae = vae.requires_grad_(False).eval()
        self.teacher = teacher.requires_grad_(False).eval() if teacher is not None else None
        self.scheduler = scheduler
        self.feature_dims = dict(feature_dims or FEATURE_DIMS)
        self.beta = float(beta)
        self.teacher_timesteps = int(teacher_timesteps)
        self.model_id = str(model_id)
        self.mixed_precision = mixed_precision
        self.timestep = nn.Parameter(torch.tensor(float(timestep)))
        self.register_buffer("empty_prompt", empty_prompt.detach().clone())
        if teacher is not None:
            self.time_features = FourierFeatures(1, 256)
            self.time_projection = nn.Linear(256, 256, bias=False)
            self.time_mapping = MappingNetwork(2, 256, 768)
            self.projection_heads = nn.ModuleDict({
                name: ProjectionHead(channels) for name, channels in self.feature_dims.items()
            })

    @property
    def device(self):
        return self.timestep.device

    def autocast(self):
        if self.mixed_precision and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        if self.teacher is not None:
            self.teacher.eval()
        return self

    @torch.no_grad()
    def encode(self, frames):
        dtype = next(self.vae.parameters()).dtype
        posterior = self.vae.encode(frames.to(device=self.device, dtype=dtype)).latent_dist
        return posterior.mode() * self.vae.config.scaling_factor

    def student_features(self, latents):
        dtype = next(self.student.parameters()).dtype
        return self.student(latents.to(dtype), self.timestep.expand(latents.shape[0]),
                            self.empty_prompt.expand(latents.shape[0], -1, -1).to(dtype))

    def project(self, features, timestep):
        denominator = max(1, self.scheduler.config.num_train_timesteps - 1)
        value = timestep.reshape(1, 1).float().to(self.device) / denominator
        condition = self.time_mapping(self.time_projection(self.time_features(value)))
        return {name: self.projection_heads[name](features[name], condition.expand(2, -1))
                for name in self.feature_dims}

    @torch.no_grad()
    def teacher_features(self, latents, timestep, noise):
        dtype = next(self.teacher.parameters()).dtype
        timesteps = timestep.expand(2)
        noisy = self.scheduler.add_noise(latents, noise, timesteps)
        return self.teacher(noisy.to(dtype), timesteps,
                            self.empty_prompt.expand(2, -1, -1).to(dtype))

    def backward_pair(self, frames, loss_scale=1.0, generator=None):
        if self.teacher is None or self.scheduler is None:
            raise RuntimeError("Training requires a frozen teacher and its noise schedule")
        if frames.shape[0] != 2:
            raise ValueError("Training expects one adjacent frame pair per microbatch")
        latents = self.encode(frames)
        with self.autocast():
            student = self.student_features(latents)
        proxies = {name: student[name].detach().requires_grad_(True) for name in self.feature_dims}
        timesteps = sample_stratified_timesteps(self.scheduler.config.num_train_timesteps,
                                                self.teacher_timesteps, self.device, generator)
        relation_target = torch.zeros(len(self.feature_dims), device=self.device)
        alignment_value = torch.zeros((), device=self.device)
        for timestep in timesteps:
            noise = shared_pair_noise(latents, generator)
            with self.autocast():
                teacher = self.teacher_features(latents, timestep, noise)
                projected = self.project(proxies, timestep)
                alignment = torch.stack([
                    1 - spatial_cosine(projected[name], teacher[name]).mean()
                    for name in self.feature_dims
                ]).mean() / self.teacher_timesteps
            with torch.no_grad():
                relation_target += torch.stack([
                    spatial_cosine(teacher[name][:1], teacher[name][1:]).squeeze(0)
                    for name in self.feature_dims
                ]) / self.teacher_timesteps
                alignment_value += alignment.detach()
            (alignment * loss_scale).backward()
            del teacher, projected, alignment
        relation_student = torch.stack([
            spatial_cosine(proxies[name][:1], proxies[name][1:]).squeeze(0)
            for name in self.feature_dims
        ])
        relation = F.mse_loss(relation_student, relation_target)
        if self.beta > 0:
            (self.beta * relation * loss_scale).backward()
        torch.autograd.backward([student[name] for name in self.feature_dims],
                                [proxies[name].grad for name in self.feature_dims])
        return {"feature_alignment": float(alignment_value), "temporal_relation": float(relation.detach()),
                "loss": float(alignment_value + self.beta * relation.detach())}

    @torch.no_grad()
    def extract(self, frames):
        with self.autocast():
            features = self.student_features(self.encode(frames))
        return frame_descriptors(features, tuple(self.feature_dims))

    def save_student(self, path, image_size=(384, 384), crop_percent=(0, 0, 0, 0),
                     training_steps=0, dataset_name=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format": "trd_student_v1", "model_id": self.model_id,
            "student": {name: value.detach().cpu() for name, value in self.student.state_dict().items()},
            "timestep": self.timestep.detach().cpu(), "empty_prompt": self.empty_prompt.detach().cpu(),
            "feature_dims": self.feature_dims, "image_size": tuple(image_size),
            "crop_percent": tuple(crop_percent), "beta": self.beta,
            "training_steps": int(training_steps), "dataset_name": dataset_name,
        }
        temporary = path.with_name(path.name + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    @classmethod
    def from_pretrained(cls, model_id="stabilityai/stable-diffusion-2-1", device="cuda",
                        beta=0.1, timestep=261.0, teacher_timesteps=5, mixed_precision=True,
                        local_files_only=False):
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        device = torch.device(device)
        dtype = torch.bfloat16 if mixed_precision and device.type == "cuda" else torch.float32
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
            local_files_only=local_files_only,
        ).to(device)
        teacher = SD21Features().to(device=device, dtype=dtype)
        teacher.load_state_dict(pipeline.unet.state_dict(), strict=True)
        del pipeline.unet
        with torch.no_grad():
            empty_prompt, _ = pipeline.encode_prompt("", device, 1, False)
        vae = pipeline.vae
        scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        student = copy.deepcopy(teacher).float().requires_grad_(True)
        model = cls(student, vae, empty_prompt, teacher, scheduler, beta=beta, timestep=timestep,
                    teacher_timesteps=teacher_timesteps, model_id=model_id,
                    mixed_precision=mixed_precision).to(device)
        del pipeline
        return model

    @classmethod
    def from_checkpoint(cls, path, device="cuda", mixed_precision=True, model_id=None,
                        local_files_only=False):
        from diffusers import AutoencoderKL

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "trd_student_v1":
            raise ValueError("Expected a student checkpoint produced by train_trd.py")
        if tuple(checkpoint["feature_dims"].items()) != tuple(FEATURE_DIMS.items()):
            raise ValueError("Checkpoint feature layers do not match the 13-layer SD2.1 extractor")
        model_id = model_id or checkpoint["model_id"]
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", local_files_only=local_files_only)
        student = SD21Features()
        student.load_state_dict(checkpoint["student"], strict=True)
        model = cls(student, vae, checkpoint["empty_prompt"], beta=checkpoint["beta"],
                    timestep=float(checkpoint["timestep"]), model_id=model_id,
                    mixed_precision=mixed_precision).to(device).eval()
        model.requires_grad_(False)
        return model, checkpoint
