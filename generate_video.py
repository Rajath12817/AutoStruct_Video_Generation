"""
End-to-end demo: natural language instruction -> final stitched video.

Stage 1-2  Parsing              parser.py          (InstructionParser)
Stage 3    Temporal scheduling  scheduler.py        (TemporalScheduler)
Stage 4    Clip retrieval       video_generator.py  (VideoGenerator)
Stage 5    Assembly             assembler.py         (VideoAssembler)

Usage:
    python generate_video.py
    python generate_video.py "wet hands then apply soap and rinse off"
    python generate_video.py "..." --strict --transition crossfade --output out/demo.mp4

Exit codes: 0 on success, 1 on any pipeline failure (see stderr for which
stage failed and why -- each failure is wrapped in PipelineError so the
cause is never a bare, unlabeled traceback).
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from env_loader import load_dotenv_file

logger = logging.getLogger("generate_video")

DEFAULT_INSTRUCTION = (
    "Turn on the tap. Wet your hands. Apply soap. Scrub your hands "
    "for 20 seconds. Rinse your hands. Dry with a clean towel."
)


class PipelineError(Exception):
    """Raised when any pipeline stage fails, with the failing stage named."""


@dataclass
class PipelineRunResult:
    instruction: str
    steps: List[str]
    timeline: List[Dict[str, Any]]
    output_path: str
    duration: float
    parser_warnings: List[str] = field(default_factory=list)
    generation_warnings: List[str] = field(default_factory=list)
    assembly_warnings: List[str] = field(default_factory=list)


def generate_video(
    instruction: str,
    output_path: str = "output/final_video.mp4",
    clips_dir: str = "clips",
    processed_dir: str = "processed_clips",
    transition: str = "cut",
    crossfade_duration: float = 0.3,
    strict: bool = False,
    show_labels: bool = True,
    trim_anchor: str = "start",
) -> PipelineRunResult:
    if not isinstance(instruction, str) or not instruction.strip():
        raise PipelineError("Instruction must be a non-empty string.")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise PipelineError("GROQ_API_KEY is not set. Add it to .env or your environment before running.")

    # Imported lazily so a bad/missing API key fails fast above, before the
    # heavier model-loading work in these constructors even starts.
    from parser import InstructionParser
    from scheduler import TemporalScheduler
    from video_generator import GenerationResult, VideoGenerator, VideoGeneratorError
    from assembler import AssemblerError, VideoAssembler
    from video_utils import detect_source_resolution

    target_resolution = detect_source_resolution(clips_dir)
    logger.info("Using target resolution %s (detected from clips in '%s').", target_resolution, clips_dir)

    logger.info("Input instruction: %r", instruction)

    try:
        parser = InstructionParser(groq_api_key=api_key)
        parse_result = parser.parse(instruction, verbose=False)
    except Exception as exc:
        raise PipelineError(f"Stage 1-2 (Parsing) failed: {exc}") from exc

    steps = [a.step for a in parse_result.actions]
    if not steps:
        raise PipelineError(
            "Stage 1-2 (Parsing) produced zero actions -- the instruction may be "
            "empty, unparseable, or entirely filtered out. Nothing to schedule or render."
        )
    logger.info("[1-2] Parsed %d action(s): %s", len(steps), steps)
    if parse_result.warnings:
        logger.warning("[1-2] Parser warnings: %s", parse_result.warnings)

    try:
        scheduler = TemporalScheduler(groq_api_key=api_key)
        timeline = scheduler.schedule(steps)
    except Exception as exc:
        raise PipelineError(f"Stage 3 (Scheduling) failed: {exc}") from exc

    if not timeline:
        raise PipelineError("Stage 3 (Scheduling) returned an empty timeline.")

    # Carry the parser's original sentence through as the on-screen caption
    # (e.g. "Turn on the tap") rather than the snake_case step name -- order
    # and count are preserved 1:1 from parsing through scheduling.
    for entry, action in zip(timeline, parse_result.actions):
        entry["label"] = action.raw.strip().rstrip(".")

    logger.info("[3] Scheduled timeline:")
    for entry in timeline:
        logger.info(
            "      %-20s %3ss -> %3ss (%ss)",
            entry["step"], entry["start"], entry["end"], entry["duration"],
        )

    try:
        generator = VideoGenerator(
            clips_dir=clips_dir, output_dir=processed_dir, strict=strict,
            target_resolution=target_resolution, trim_anchor=trim_anchor,
        )
        generation: GenerationResult = generator.generate(timeline)
    except VideoGeneratorError as exc:
        raise PipelineError(f"Stage 4 (Clip generation) failed: {exc}") from exc
    except Exception as exc:
        raise PipelineError(f"Stage 4 (Clip generation) failed unexpectedly: {exc}") from exc

    logger.info("[4] Prepared %d clip(s).", len(generation.clips))
    if generation.warnings:
        for w in generation.warnings:
            logger.warning("[4] %s", w)

    try:
        # generation.clips is 1:1 with timeline (same order, same count --
        # generate() never drops or reorders entries), so position-based
        # zipping is safe even when the same step name repeats.
        labels = [c.label for c in generation.clips] if show_labels else None

        assembler = VideoAssembler(output_path=output_path, target_resolution=target_resolution)
        assembly = assembler.assemble(
            generation.clip_paths, transition=transition, crossfade_duration=crossfade_duration,
            labels=labels,
        )
    except AssemblerError as exc:
        raise PipelineError(f"Stage 5 (Assembly) failed: {exc}") from exc
    except Exception as exc:
        raise PipelineError(f"Stage 5 (Assembly) failed unexpectedly: {exc}") from exc

    logger.info("[5] Final video written to: %s (%.1fs)", assembly.output_path, assembly.duration)

    result = PipelineRunResult(
        instruction=instruction,
        steps=steps,
        timeline=timeline,
        output_path=assembly.output_path,
        duration=assembly.duration,
        parser_warnings=parse_result.warnings,
        generation_warnings=generation.warnings,
        assembly_warnings=assembly.warnings,
    )

    report_path = os.path.splitext(output_path)[0] + "_report.json"
    with open(report_path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    logger.info("Run report saved to: %s", report_path)

    return result


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Instruction -> stitched video pipeline.")
    p.add_argument("instruction", nargs="?", default=DEFAULT_INSTRUCTION, help="Natural language instruction.")
    p.add_argument("--output", default="output/final_video.mp4", help="Path for the final mp4.")
    p.add_argument("--clips-dir", default="clips", help="Directory of pre-recorded per-action clips.")
    p.add_argument("--processed-dir", default="processed_clips", help="Directory for Stage 4 intermediate clips.")
    p.add_argument("--transition", choices=["cut", "crossfade"], default="cut")
    p.add_argument("--crossfade-duration", type=float, default=0.3)
    p.add_argument(
        "--strict", action="store_true",
        help="Fail immediately on any missing/broken clip instead of degrading to a placeholder.",
    )
    p.add_argument(
        "--no-labels", action="store_true",
        help="Don't burn the current step's text onto the top of each clip.",
    )
    p.add_argument(
        "--trim-anchor", choices=["start", "center", "end"], default="start",
        help="Where in a longer-than-needed source clip to take the trimmed window from. "
             "'start' suits recorded footage where the motion begins immediately; 'end' suits "
             "clips (often generative-model output) whose meaningful moment builds up over time.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    load_dotenv_file()

    try:
        result = generate_video(
            args.instruction,
            output_path=args.output,
            clips_dir=args.clips_dir,
            processed_dir=args.processed_dir,
            transition=args.transition,
            crossfade_duration=args.crossfade_duration,
            strict=args.strict,
            show_labels=not args.no_labels,
            trim_anchor=args.trim_anchor,
        )
    except PipelineError as exc:
        print(f"PIPELINE FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Done. Final video: {result.output_path} ({result.duration:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
