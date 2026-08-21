"""
Shared clip-normalization logic for Stage 4 (video_generator.py) and
Stage 5 (assembler.py), so resolution/fps handling is identical no matter
which stage touches a clip or whether it came from the local clip bank,
a plugged-in generative model, or a caller of VideoAssembler directly.
"""

import os
from typing import Tuple

from moviepy import VideoFileClip, vfx

DEFAULT_RESOLUTION: Tuple[int, int] = (1280, 720)
DEFAULT_FPS = 24
SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv")


def normalize_clip(clip, target_resolution: Tuple[int, int] = DEFAULT_RESOLUTION, target_fps: int = DEFAULT_FPS):
    """
    Letterbox-fit `clip` into target_resolution (preserve aspect ratio, pad
    with black bars rather than stretching) and fix its fps. Stretching to a
    fixed size would distort footage whose source aspect ratio differs from
    the target -- a real risk once Stage 4 can draw from mixed sources
    (your recorded clips vs. a generative model's output resolution).
    """
    target_w, target_h = target_resolution
    src_w, src_h = clip.size
    scale = min(target_w / src_w, target_h / src_h)
    new_size = (max(1, round(src_w * scale)), max(1, round(src_h * scale)))

    clip = clip.with_effects([vfx.Resize(new_size)])
    if new_size != (target_w, target_h):
        left = (target_w - new_size[0]) // 2
        top = (target_h - new_size[1]) // 2
        clip = clip.with_effects([vfx.Margin(
            left=left, right=target_w - new_size[0] - left,
            top=top, bottom=target_h - new_size[1] - top,
            color=(0, 0, 0),
        )])
    return clip.with_fps(target_fps)


def detect_source_resolution(clips_dir: str, default: Tuple[int, int] = DEFAULT_RESOLUTION) -> Tuple[int, int]:
    """
    Peek at the first readable clip in clips_dir to infer whether footage is
    portrait or landscape, so the assembled video isn't padded with wasted
    black bars just because a hardcoded default didn't match how the clips
    were actually filmed (e.g. phone video is portrait by default).
    Returns a 1280-long-edge resolution matching the source's orientation
    and aspect ratio, or `default` if clips_dir is empty/unreadable.
    """
    if not os.path.isdir(clips_dir):
        return default
    for fname in sorted(os.listdir(clips_dir)):
        if os.path.splitext(fname)[1].lower() not in SUPPORTED_EXTENSIONS:
            continue
        path = os.path.join(clips_dir, fname)
        try:
            clip = VideoFileClip(path)
            src_w, src_h = clip.size
            clip.close()
        except Exception:
            continue
        if src_h > src_w:  # portrait
            target_h = 1280
            target_w = round(target_h * src_w / src_h)
        else:  # landscape or square
            target_w = 1280
            target_h = round(target_w * src_h / src_w)
        target_w -= target_w % 2  # even dimensions -- required by libx264
        target_h -= target_h % 2
        return (target_w, target_h)
    return default
