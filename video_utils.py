"""
Shared clip-normalization logic for Stage 4 (video_generator.py) and
Stage 5 (assembler.py), so resolution/fps handling is identical no matter
which stage touches a clip or whether it came from the local clip bank,
a plugged-in generative model, or a caller of VideoAssembler directly.
"""

from typing import Tuple

from moviepy import vfx

DEFAULT_RESOLUTION: Tuple[int, int] = (1280, 720)
DEFAULT_FPS = 24


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
