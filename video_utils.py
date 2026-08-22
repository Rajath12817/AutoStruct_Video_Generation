"""
Shared clip-normalization logic for Stage 4 (video_generator.py) and
Stage 5 (assembler.py), so resolution/fps handling is identical no matter
which stage touches a clip or whether it came from the local clip bank,
a plugged-in generative model, or a caller of VideoAssembler directly.
"""

import logging
import os
from typing import Optional, Tuple

from moviepy import TextClip, VideoFileClip, vfx

logger = logging.getLogger("video_utils")

DEFAULT_RESOLUTION: Tuple[int, int] = (1280, 720)
DEFAULT_FPS = 24
SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv")
ASPECT_RATIO_OUTLIER_TOLERANCE = 0.15  # 15% off the chosen ratio triggers a warning

# Best-effort bold font for step-name captions -- moviepy/Pillow renders fine
# without one (falls back to a built-in font), this just looks nicer when available.
_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)


def _find_caption_font() -> Optional[str]:
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


CAPTION_FONT = _find_caption_font()


def build_caption_bar(text: str, target_resolution: Tuple[int, int]):
    """A black bar + white text spanning the top of target_resolution, sized
    to that resolution so it can be composited directly at position (0, 0)."""
    target_w, target_h = target_resolution
    bar_h = max(40, round(target_h * 0.08))
    return TextClip(
        font=CAPTION_FONT,
        text=text,
        font_size=max(18, round(bar_h * 0.45)),
        color="white",
        bg_color="black",
        size=(target_w, bar_h),
        method="caption",
        text_align="center",
        horizontal_align="center",
        vertical_align="center",
    )


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
    Inspect every readable clip in clips_dir to infer a representative
    orientation and aspect ratio, so the assembled video isn't padded with
    wasted black bars just because a hardcoded default didn't match how the
    clips were actually filmed (e.g. phone video is portrait by default).

    Uses a majority vote across ALL clips, not just one -- once the clip bank
    mixes sources (your recorded phone footage alongside a generative model's
    output, which is common at a different native aspect ratio), trusting a
    single clip would let whichever one happens to sort first silently decide
    the shape for everyone else. Any clip whose aspect ratio disagrees with
    the chosen one by more than 15% gets logged so a real mismatch isn't
    silently swallowed as "it's fine, it'll get letterboxed."

    Returns a 1280-long-edge resolution, or `default` if clips_dir is
    empty/unreadable.
    """
    if not os.path.isdir(clips_dir):
        return default

    sizes = []  # (filename, width, height)
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
        sizes.append((fname, src_w, src_h))

    if not sizes:
        return default

    portrait_count = sum(1 for _, w, h in sizes if h > w)
    use_portrait = portrait_count > len(sizes) - portrait_count

    matching_ratios = sorted(
        (w / h) for _, w, h in sizes if (h > w) == use_portrait
    )
    median_ratio = matching_ratios[len(matching_ratios) // 2]

    outliers = [
        f"{fname} ({w}x{h}, ratio {w / h:.2f})"
        for fname, w, h in sizes
        if abs((w / h) - median_ratio) / median_ratio > ASPECT_RATIO_OUTLIER_TOLERANCE
    ]
    if outliers:
        logger.warning(
            "%d clip(s) in '%s' have a notably different aspect ratio than the "
            "rest and will be letterboxed more heavily: %s",
            len(outliers), clips_dir, "; ".join(outliers),
        )

    if use_portrait:
        target_h = 1280
        target_w = round(target_h * median_ratio)
    else:
        target_w = 1280
        target_h = round(target_w / median_ratio)
    target_w -= target_w % 2  # even dimensions -- required by libx264
    target_h -= target_h % 2
    return (target_w, target_h)
