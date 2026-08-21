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
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Tuple

from moviepy import VideoFileClip, concatenate_videoclips, vfx

from video_utils import DEFAULT_FPS, DEFAULT_RESOLUTION, normalize_clip

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
    ) -> AssemblyResult:
        if not clip_paths:
            raise AssemblerError("No clips to assemble.")
        if transition not in VALID_TRANSITIONS:
            raise AssemblerError(f"Unknown transition '{transition}' (use one of {VALID_TRANSITIONS}).")
        if transition == "crossfade" and crossfade_duration <= 0:
            raise AssemblerError(f"crossfade_duration must be positive, got {crossfade_duration}")

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

            if transition == "crossfade" and len(normalized) > 1:
                min_dur = min(c.duration for c in normalized)
                safe_crossfade = min(crossfade_duration, min_dur * MIN_CROSSFADE_FRACTION)
                if safe_crossfade < crossfade_duration:
                    warnings.append(
                        f"crossfade_duration clamped from {crossfade_duration}s to {safe_crossfade:.2f}s "
                        f"(shortest clip is {min_dur:.2f}s)."
                    )
                final = self._assemble_crossfade(normalized, safe_crossfade)
            else:
                final = concatenate_videoclips(normalized, method="chain")

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
