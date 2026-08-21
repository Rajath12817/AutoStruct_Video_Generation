"""
Stage 4 — Clip Retrieval / Generation
======================================
Consumes the scheduler's timeline (list of {step, start, end, duration})
and produces one normalized, correctly-timed video clip per timeline entry.

For each step, in order:
  1. Look up a pre-recorded clip in the clip bank (clips/<step>.<ext>), by
     exact filename first, then by keyword overlap (fuzzy match) -- the
     parser has no fixed action vocabulary, so "apply soap to your hands"
     naturally becomes the step apply_soap_to_your_hands, not apply_soap.
     Since the clip bank IS this project's controlled action space, fuzzy
     matching against it is what actually closes that gap.
  2. If missing or unreadable, and a `fallback_generator` is configured
     (e.g. a real text-to-video model), use that instead.
  3. If that's also unavailable or fails, fall back to a labeled placeholder
     clip so the run still completes with the right number/length of clips.
  4. Trim (if longer than needed) or loop (if shorter) to the scheduler's
     assigned duration, then letterbox-normalize resolution/fps so every
     clip -- regardless of source -- can be concatenated cleanly in Stage 5.

Malformed timeline data (missing 'step', non-dict entries) always raises --
there's no reasonable fallback for "I don't know what action this is." A
missing or broken *clip* is a different, recoverable class of problem.

`strict=True` disables all graceful degradation: any missing/broken clip or
failed fallback raises immediately instead of substituting a placeholder.
Use it in CI/automated testing where a silently-degraded run is worse than
a loud failure; leave it off for demos where "always produce a video" matters.

Clip bank convention
---------------------
Put one file per action in clips/, named after the parser's `step` value:
    clips/turn_on_tap.mp4
    clips/wet_hands.mp4
    clips/apply_soap.mp4
    clips/lather_hands.mp4
    clips/scrub_hands.mp4
    clips/rinse_hands.mp4
    clips/dry_hands.mp4
Same person, same static camera, same background across all of them --
that consistency has to come from how they're filmed, not from code.

Plugging in a real text-to-video model later
---------------------------------------------
Pass `fallback_generator` to VideoGenerator:

    def videocrafter_generate(step: str, duration: int) -> str:
        # call VideoCrafter2 / AnimateDiff / a hosted T2V API here,
        # conditioned on a fixed reference frame for consistency,
        # return the path to the rendered mp4
        ...

    VideoGenerator(fallback_generator=videocrafter_generate)

Its output is re-normalized through the exact same pipeline as clip-bank
footage, so it always produces the correct duration/resolution/fps -- and
if it fails, generate() degrades to a placeholder instead of crashing the
whole run (unless strict=True).
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from moviepy import ColorClip, VideoFileClip, vfx

from video_utils import DEFAULT_FPS, DEFAULT_RESOLUTION, SUPPORTED_EXTENSIONS, normalize_clip

logger = logging.getLogger("video_generator")

MIN_DURATION_S = 1
MAX_DURATION_S = 30  # sanity cap against a runaway/hallucinated scheduler duration

# The parser has no fixed action vocabulary -- it extracts verb+object
# straight from the sentence, so "apply soap to your hands" becomes
# apply_soap_to_your_hands, not apply_soap. Since the clip bank *is* the
# controlled action space for this project, an exact-filename-only lookup
# would silently placeholder every real clip you record. These filler
# words get dropped before matching so token overlap still lines up.
FUZZY_MATCH_STOPWORDS = {
    "a", "an", "the", "your", "my", "his", "her", "their", "to", "for",
    "with", "of", "in", "seconds", "second", "minutes", "minute", "clean",
    "some", "off", "it", "them",
}


def _tokenize(step: str) -> set:
    return {t for t in step.strip().lower().split("_") if t and t not in FUZZY_MATCH_STOPWORDS}


class VideoGeneratorError(Exception):
    """Base class for Stage 4 errors."""


class ClipReadError(VideoGeneratorError):
    """A source clip exists but could not be opened or decoded."""


class ClipWriteError(VideoGeneratorError):
    """A processed clip failed to render/write to disk."""


class TimelineValidationError(VideoGeneratorError):
    """The scheduler's timeline was malformed -- always fatal, never degraded."""


@dataclass
class ClipResult:
    step: str
    path: str
    duration: float
    source: str  # "library" | "fallback_generator" | "placeholder" | "placeholder_after_error"
    label: str = ""
    is_fallback: bool = False


@dataclass
class GenerationResult:
    clips: List[ClipResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def clip_paths(self) -> List[str]:
        return [c.path for c in self.clips]


def _safe_filename(step: str) -> str:
    """Collapse a step name into a filesystem-safe component (blocks path traversal / illegal chars)."""
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", step.strip())
    name = name.strip("._") or "step"
    return name[:100]


class ClipLibrary:
    """Indexes clips/<step_name>.<ext> files by step name (case-insensitive)."""

    def __init__(self, clips_dir: str = "clips"):
        self.clips_dir = clips_dir
        self.index: Dict[str, str] = {}
        self._build_index()

    def _build_index(self) -> None:
        if not os.path.isdir(self.clips_dir):
            logger.warning("Clip directory '%s' not found -- no clips indexed yet.", self.clips_dir)
            return
        for fname in sorted(os.listdir(self.clips_dir)):
            name, ext = os.path.splitext(fname)
            if ext.lower() not in SUPPORTED_EXTENSIONS:
                continue
            key = name.strip().lower()
            full_path = os.path.join(self.clips_dir, fname)
            if key in self.index:
                logger.warning(
                    "Duplicate clip name '%s' -- keeping '%s', ignoring '%s'.",
                    key, self.index[key], full_path,
                )
                continue
            self.index[key] = full_path
        logger.info("Indexed %d clip(s) from '%s'.", len(self.index), self.clips_dir)

    def get(self, step: str) -> Optional[str]:
        path, _ = self.resolve(step)
        return path

    def has(self, step: str) -> bool:
        return self.get(step) is not None

    def resolve(self, step: str) -> Tuple[Optional[str], str]:
        """Returns (path, match_type) where match_type is 'exact', 'fuzzy', or 'none'."""
        key = step.strip().lower()
        if key in self.index:
            return self.index[key], "exact"

        query_tokens = _tokenize(key)
        best_key, best_score = None, 0.0
        for lib_key in self.index:
            lib_tokens = _tokenize(lib_key)
            # require every token of the canonical action name to be present
            # in the step -- "hands" alone must not match wet_hands AND
            # rinse_hands both; only a step containing "wet"+"hands" (or
            # "rinse"+"hands") should qualify.
            if not lib_tokens or not query_tokens or not lib_tokens.issubset(query_tokens):
                continue
            score = len(lib_tokens) / len(query_tokens)  # prefer the tightest/most specific match
            if score > best_score:
                best_key, best_score = lib_key, score

        if best_key is not None:
            logger.info("Fuzzy-matched step '%s' -> clip '%s' (keyword overlap).", key, best_key)
            return self.index[best_key], "fuzzy"
        return None, "none"


class VideoGenerator:
    """Turns a scheduled timeline into ordered, normalized clip files for Stage 5."""

    def __init__(
        self,
        clips_dir: str = "clips",
        output_dir: str = "processed_clips",
        target_resolution: Tuple[int, int] = DEFAULT_RESOLUTION,
        target_fps: int = DEFAULT_FPS,
        fallback_generator: Optional[Callable[[str, int], str]] = None,
        strict: bool = False,
        max_duration_s: int = MAX_DURATION_S,
    ):
        if target_resolution[0] <= 0 or target_resolution[1] <= 0:
            raise ValueError(f"target_resolution must be positive, got {target_resolution}")
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        if max_duration_s < MIN_DURATION_S:
            raise ValueError(f"max_duration_s must be >= {MIN_DURATION_S}, got {max_duration_s}")

        self.library = ClipLibrary(clips_dir)
        self.output_dir = output_dir
        self.target_resolution = target_resolution
        self.target_fps = target_fps
        self.fallback_generator = fallback_generator
        self.strict = strict
        self.max_duration_s = max_duration_s
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self, timeline: List[Dict[str, Any]]) -> GenerationResult:
        if not isinstance(timeline, list):
            raise TimelineValidationError(f"timeline must be a list, got {type(timeline).__name__}")
        if not timeline:
            raise TimelineValidationError("timeline is empty -- nothing to generate.")

        result = GenerationResult()
        for i, entry in enumerate(timeline):
            clip_result = self._generate_one(i, entry)
            result.clips.append(clip_result)
            if clip_result.is_fallback:
                result.warnings.append(
                    f"Step '{clip_result.step}' (position {i}) used '{clip_result.source}' instead of a library clip."
                )
            elif clip_result.source == "library_fuzzy":
                result.warnings.append(
                    f"Step '{clip_result.step}' (position {i}) matched a clip via fuzzy keyword "
                    f"matching, not an exact filename match -- verify it picked the right clip."
                )
        return result

    # ── per-step generation, with layered fallback ──────────────────────

    def _generate_one(self, i: int, entry: Any) -> ClipResult:
        step, duration, label = self._validate_entry(i, entry)
        safe_name = _safe_filename(step)
        out_path = os.path.join(self.output_dir, f"{i:03d}_{safe_name}.mp4")

        source_path, match_type = self.library.resolve(step)
        if source_path is not None:
            try:
                source_label = "library" if match_type == "exact" else "library_fuzzy"
                return self._render(step, source_path, duration, out_path, source_label=source_label, label=label)
            except VideoGeneratorError as exc:
                if self.strict:
                    raise
                logger.error("Library clip for '%s' failed (%s); trying fallback.", step, exc)

        if self.fallback_generator is not None:
            try:
                gen_path = self.fallback_generator(step, duration)
                if not gen_path or not os.path.isfile(gen_path):
                    raise VideoGeneratorError(f"fallback_generator for '{step}' returned no usable file")
                return self._render(step, gen_path, duration, out_path, source_label="fallback_generator", label=label)
            except VideoGeneratorError as exc:
                if self.strict:
                    raise
                logger.error("fallback_generator for '%s' failed (%s); using placeholder.", step, exc)
            except Exception as exc:
                if self.strict:
                    raise VideoGeneratorError(f"fallback_generator raised for '{step}': {exc}") from exc
                logger.error("fallback_generator for '%s' raised %s; using placeholder.", step, exc)

        if self.strict:
            raise VideoGeneratorError(f"no clip available for step '{step}' and strict mode is on")

        path = self._placeholder_clip(step, duration, index=i)
        source_label = "placeholder" if source_path is None and self.fallback_generator is None else "placeholder_after_error"
        return ClipResult(step=step, path=path, duration=duration, source=source_label, label=label, is_fallback=True)

    def _validate_entry(self, i: int, entry: Any) -> Tuple[str, int, str]:
        if not isinstance(entry, dict):
            raise TimelineValidationError(f"timeline[{i}] must be a dict, got {type(entry).__name__}")
        step = entry.get("step")
        if not step or not isinstance(step, str):
            raise TimelineValidationError(f"timeline[{i}] missing a valid 'step' string")
        duration = self._clamp_duration(entry.get("duration", 3))
        label = entry.get("label")
        if not label or not isinstance(label, str):
            label = step.replace("_", " ").strip().title()
        return step, duration, label

    def _clamp_duration(self, raw_duration: Any) -> int:
        try:
            duration = int(round(float(raw_duration)))
        except (TypeError, ValueError):
            logger.warning("Invalid duration %r -- defaulting to 3s.", raw_duration)
            duration = 3
        clamped = max(MIN_DURATION_S, min(duration, self.max_duration_s))
        if clamped != duration:
            logger.warning(
                "Duration %ss out of [%s,%s]s range -- clamped to %ss.",
                duration, MIN_DURATION_S, self.max_duration_s, clamped,
            )
        return clamped

    # ── rendering ─────────────────────────────────────────────────────

    def _render(
        self, step: str, source_path: str, duration: int, out_path: str, source_label: str, label: str,
    ) -> ClipResult:
        try:
            source_clip = VideoFileClip(source_path)
        except Exception as exc:
            raise ClipReadError(f"could not open '{source_path}' for step '{step}': {exc}") from exc

        try:
            if not source_clip.duration or source_clip.duration <= 0:
                raise ClipReadError(f"clip '{source_path}' for step '{step}' has zero/invalid duration")

            clip = self._fit_duration(source_clip, duration)
            clip = normalize_clip(clip, self.target_resolution, self.target_fps)
            try:
                clip.write_videofile(
                    out_path, fps=self.target_fps, codec="libx264", audio=False, logger=None,
                )
            except Exception as exc:
                raise ClipWriteError(f"failed writing '{out_path}' for step '{step}': {exc}") from exc
            finally:
                clip.close()
        finally:
            source_clip.close()

        return ClipResult(
            step=step, path=out_path, duration=duration, label=label,
            source=source_label, is_fallback=(source_label not in ("library", "library_fuzzy")),
        )

    def _fit_duration(self, clip, duration: int):
        if clip.duration >= duration:
            return clip.subclipped(0, duration)
        return clip.with_effects([vfx.Loop(duration=duration)])

    def _placeholder_clip(self, step: str, duration: int, index: int) -> str:
        logger.warning("No clip for '%s' -- generating placeholder (%ss).", step, duration)
        color = self._color_for_step(step)
        clip = ColorClip(size=self.target_resolution, color=color).with_duration(duration)
        clip = clip.with_fps(self.target_fps)
        safe_name = _safe_filename(step)
        path = os.path.join(self.output_dir, f"{index:03d}_placeholder_{safe_name}.mp4")
        try:
            clip.write_videofile(path, fps=self.target_fps, codec="libx264", audio=False, logger=None)
        except Exception as exc:
            raise ClipWriteError(f"failed writing placeholder for step '{step}': {exc}") from exc
        finally:
            clip.close()
        return path

    @staticmethod
    def _color_for_step(step: str) -> Tuple[int, int, int]:
        h = abs(hash(step))
        return (80 + h % 150, 80 + (h // 7) % 150, 80 + (h // 13) % 150)
