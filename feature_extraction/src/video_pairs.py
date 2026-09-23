from bisect import bisect_right
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_key(path):
    return tuple((0, int(part)) if part.isdigit() else (1, part.lower())
                 for part in re.split(r"(\d+)", str(path)))


def discover_videos(dataset_dir):
    root = Path(dataset_dir)
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = [path for path in root.rglob("*") if path.is_file()]
    videos = [path for path in files if path.suffix.lower() in VIDEO_EXTENSIONS]
    frame_folders = {path.parent for path in files if path.suffix.lower() in IMAGE_EXTENSIONS}
    return sorted(videos + list(frame_folders), key=natural_key)


def validate_preprocessing(image_size, crop_percent):
    if len(image_size) != 2 or any(value < 64 or value % 64 for value in image_size):
        raise ValueError("Image height and width must be positive multiples of 64")
    if len(crop_percent) != 4 or any(not 0 <= value < 1 for value in crop_percent):
        raise ValueError("Crop percentages must contain four values in [0, 1)")
    left, right, top, bottom = crop_percent
    if left + right >= 1 or top + bottom >= 1:
        raise ValueError("Crop percentages must retain a nonempty image")


def frame_tensor(rgb, image_size, crop_percent):
    height, width = rgb.shape[:2]
    left, right, top, bottom = crop_percent
    image = Image.fromarray(rgb).crop((int(width * left), int(height * top),
                                      int(width * (1 - right)), int(height * (1 - bottom))))
    image = image.resize((image_size[1], image_size[0]), Image.Resampling.BICUBIC)
    pixels = np.array(image, dtype=np.float32, copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).div_(127.5).sub_(1)


class VideoFrames:
    def __init__(self, path):
        self.path = Path(path)
        self.images = None
        self.fps = None
        if self.path.is_dir():
            self.images = sorted((path for path in self.path.iterdir()
                                  if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()),
                                 key=natural_key)
            self.frame_count = len(self.images)
        elif self.path.is_file():
            capture = cv2.VideoCapture(str(self.path))
            try:
                if not capture.isOpened():
                    raise OSError(f"Cannot open video: {self.path}")
                self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = float(capture.get(cv2.CAP_PROP_FPS))
            finally:
                capture.release()
        else:
            raise FileNotFoundError(self.path)
        if self.frame_count < 1:
            raise ValueError(f"No decodable frames in {self.path}")

    def read(self, indices):
        indices = [int(index) for index in indices]
        if any(index < 0 or index >= self.frame_count for index in indices):
            raise IndexError("Frame index lies outside the video")
        if self.images is not None:
            result = []
            for index in indices:
                with Image.open(self.images[index]) as image:
                    result.append(np.array(image.convert("RGB"), copy=True))
            return result
        capture = cv2.VideoCapture(str(self.path))
        result = []
        try:
            previous = -2
            for index in indices:
                if index != previous + 1:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                success, image = capture.read()
                if not success:
                    raise OSError(f"Cannot decode frame {index} from {self.path}")
                result.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                previous = index
        finally:
            capture.release()
        return result


class AdjacentFramePairs(Dataset):
    def __init__(self, video_paths, image_size=(384, 384), crop_percent=(0, 0, 0, 0)):
        validate_preprocessing(image_size, crop_percent)
        self.image_size = tuple(image_size)
        self.crop_percent = tuple(crop_percent)
        self.videos = [VideoFrames(path) for path in video_paths]
        self.videos = [video for video in self.videos if video.frame_count >= 2]
        self.cumulative_pairs = np.cumsum([video.frame_count - 1 for video in self.videos]).tolist()
        if not self.cumulative_pairs:
            raise ValueError("At least one video with two consecutive frames is required")

    def __len__(self):
        return self.cumulative_pairs[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        video_index = bisect_right(self.cumulative_pairs, index)
        offset = self.cumulative_pairs[video_index - 1] if video_index else 0
        start = int(index - offset)
        frames = self.videos[video_index].read([start, start + 1])
        return torch.stack([frame_tensor(frame, self.image_size, self.crop_percent) for frame in frames])
