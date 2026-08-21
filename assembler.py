"""
Stage 5 — Sequential Assembly
==============================
Concatenates ordered clips (from video_generator.py, or any other source)
into one final video, in timeline order.

VideoAssembler re-validates and re-normalizes every input clip itself --
it does not assume Stage 4 already did so. That makes it safe to call
directly with output from a different source later (e.g. a generative
model wired straight in), without silently producing a distorted or
inconsistent final video.

Default transition is a hard cut ("cut"), which is the correct choice for
this project's core constraint -- only one event may be active at any
time. "crossfade" is offered for a smoother-looking demo, but it briefly
overlaps two actions and therefore breaks strict non-overlap; use it only
if that trade-off is acceptable.

Optional `labels` are captioned against clips' actual positions in the
*final* assembled timeline (computed here, accounting for crossfade
overlap), not baked into each clip beforehand. Captions must hard-cut even
when a crossfade blends the video underneath -- overlaying two different
captions on top of each other during a blend renders as unreadable
double-exposed text, so caption timing is deliberately independent of
video-transition timing.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from moviepy import CompositeVideoClip, VideoFileClip, concatenate_videoclips, vfx

from video_utils import DEFAULT_FPS, DEFAULT_RESOLUTION, build_caption_bar, normalize_clip

logger = logging.getLogger("assembler")

VALID_TRANSITIONS = ("cut", "crossfade")
MIN_CROSSFADE_FRACTION = 0.4  # crossfade never eats more than this fraction of the shortest clip


class AssemblerError(Exception):
    """Raised when Stage 5 cannot produce a final video."""


@dataclass
class AssemblyResult:
    output_path: str
    duration: float
    clip_count: int
    warnings: List[str] = field(default_factory=list)


class VideoAssembler:
    def __init__(
        self,
        output_path: str = "output/final_video.mp4",
        target_resolution: Tuple[int, int] = DEFAULT_RESOLUTION,
        fps: int = DEFAULT_FPS,
    ):
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if target_resolution[0] <= 0 or target_resolution[1] <= 0:
            raise ValueError(f"target_resolution must be positive, got {target_resolution}")

        self.output_path = output_path
        self.target_resolution = target_resolution
        self.fps = fps
        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def assemble(
        self,
        clip_paths: List[str],
        transition: str = "cut",
        crossfade_duration: float = 0.3,
        labels: Optional[List[str]] = None,
    ) -> AssemblyResult:
        if not clip_paths:
            raise AssemblerError("No clips to assemble.")
        if transition not in VALID_TRANSITIONS:
            raise AssemblerError(f"Unknown transition '{transition}' (use one of {VALID_TRANSITIONS}).")
        if transition == "crossfade" and crossfade_duration <= 0:
            raise AssemblerError(f"crossfade_duration must be positive, got {crossfade_duration}")
        if labels is not None and len(labels) != len(clip_paths):
            raise AssemblerError(
                f"labels length ({len(labels)}) must match clip_paths length ({len(clip_paths)})"
            )

        missing = [p for p in clip_paths if not os.path.isfile(p)]
        if missing:
            raise AssemblerError(f"{len(missing)} clip path(s) do not exist: {missing}")

        warnings: List[str] = []
        opened: List[VideoFileClip] = []
        final = None
        try:
            normalized = []
            for path in clip_paths:
                try:
                    raw = VideoFileClip(path)
                except Exception as exc:
                    raise AssemblerError(f"could not open clip '{path}': {exc}") from exc
                opened.append(raw)

                if not raw.duration or raw.duration <= 0:
                    raise AssemblerError(f"clip '{path}' has zero/invalid duration")

                if tuple(raw.size) != tuple(self.target_resolution):
                    warnings.append(f"'{path}' was {tuple(raw.size)}, not {self.target_resolution} -- re-normalized.")
                normalized.append(normalize_clip(raw, self.target_resolution, self.fps))

            used_crossfade = 0.0
            if transition == "crossfade" and len(normalized) > 1:
                min_dur = min(c.duration for c in normalized)
                used_crossfade = min(crossfade_duration, min_dur * MIN_CROSSFADE_FRACTION)
                if used_crossfade < crossfade_duration:
                    warnings.append(
                        f"crossfade_duration clamped from {crossfade_duration}s to {used_crossfade:.2f}s "
                        f"(shortest clip is {min_dur:.2f}s)."
                    )
                final = self._assemble_crossfade(normalized, used_crossfade)
            else:
                final = concatenate_videoclips(normalized, method="chain")

            if labels:
                final = self._overlay_labels(final, normalized, labels, used_crossfade)

            logger.info(
                "Writing final video -> %s (%.1fs, %d clip(s), transition=%s)",
                self.output_path, final.duration, len(clip_paths), transition,
            )
            try:
                final.write_videofile(
                    self.output_path, fps=self.fps, codec="libx264", audio=False, logger=None,
                )
            except Exception as exc:
                if os.path.isfile(self.output_path):
                    try:
                        os.remove(self.output_path)  # don't leave a corrupt/partial file behind
                    except OSError:
                        pass
                raise AssemblerError(f"failed writing final video '{self.output_path}': {exc}") from exc

            return AssemblyResult(
                output_path=self.output_path,
                duration=final.duration,
                clip_count=len(clip_paths),
                warnings=warnings,
            )
        finally:
            if final is not None:
                final.close()
            for c in opened:
                c.close()

    def _assemble_crossfade(self, clips: List[VideoFileClip], crossfade_duration: float):
        faded = [clips[0]] + [
            c.with_effects([vfx.CrossFadeIn(crossfade_duration)]) for c in clips[1:]
        ]
        return concatenate_videoclips(faded, method="compose", padding=-crossfade_duration)

    def _overlay_labels(self, final, clips: List[VideoFileClip], labels: List[str], crossfade_duration: float):
        """Positions one caption per clip against its actual start time in the
        final timeline (mirroring how concatenate_videoclips placed it), so
        captions hard-cut cleanly regardless of what the video is doing."""
        padding = -crossfade_duration if crossfade_duration else 0.0
        starts: List[float] = [0.0]
        for c in clips[:-1]:
            starts.append(starts[-1] + c.duration + padding)

        captions = []
        for label, start, clip in zip(labels, starts, clips):
            if not label:
                continue
            try:
                caption = build_caption_bar(label, self.target_resolution)
            except Exception as exc:
                logger.warning("Could not render caption '%s' (%s) -- skipping.", label, exc)
                continue
            captions.append(caption.with_duration(clip.duration).with_start(start).with_position(("center", 0)))

        if not captions:
            return final
        return CompositeVideoClip([final] + captions, size=self.target_resolution).with_duration(final.duration)
