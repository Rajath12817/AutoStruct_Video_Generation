"""
Performance benchmark harness (Phase III deliverable).

Runs the pipeline across a fixed set of representative handwashing
instructions -- spanning clean input, noisy/misspelled input, informal
slang, numbered-list style, and mixed tense -- and records per-stage
latency plus clip-resolution outcomes (exact match / fuzzy match /
placeholder). Produces a JSON results file used for the performance-
analysis and experimental-results report.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from env_loader import load_dotenv_file

load_dotenv_file()
API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GROQ_API_KEY before running the benchmark.")

from parser import InstructionParser
from scheduler import TemporalScheduler
from video_generator import VideoGenerator
from assembler import VideoAssembler

BENCHMARK_INSTRUCTIONS = [
    ("clean_well_formed",
     "Turn on the tap. Wet your hands. Apply soap. Scrub your hands for 20 seconds. "
     "Rinse your hands. Dry with a clean towel."),
    ("who_7step_informal",
     "wet hands with water then aply soaap rub palms together interlok fingers scrub "
     "backs of hands rubb thumbs rince with water and drie with towel"),
    ("mixed_tense_typos",
     "first trun on the tap then wets your hand aply some soaap then she scrubs and "
     "rinced off finally dry hand"),
    ("informal_slang",
     "yo open tap get hands wet put soap on hands rub rub rub get the soap off then dry it"),
    ("single_run_on",
     "turn on tap wet hands apply soap lather scrub rinse dry"),
    ("numbered_list",
     "1 turn on tap 2 wet your hands 3 apply soap 4 rub hands 5 rinse off soap "
     "6 turn off tap 7 dry hands with towel"),
]


@dataclass
class StageTiming:
    parse_s: float = 0.0
    schedule_s: float = 0.0
    generate_s: float = 0.0
    assemble_s: float = 0.0
    total_s: float = 0.0


@dataclass
class RunResult:
    label: str
    instruction: str
    timing: StageTiming
    num_actions: int
    steps: List[str]
    clip_sources: Dict[str, int]  # {"library": n, "library_fuzzy": n, "placeholder": n, ...}
    final_duration_s: float
    parser_warning_count: int
    generation_warning_count: int
    status: str
    error: str = ""


def run_once(label: str, instruction: str, output_dir: str, transition: str = "cut") -> RunResult:
    timing = StageTiming()
    t_start = time.perf_counter()
    try:
        parser = InstructionParser(groq_api_key=API_KEY)
        scheduler = TemporalScheduler(groq_api_key=API_KEY)

        t0 = time.perf_counter()
        parse_result = parser.parse(instruction, verbose=False)
        timing.parse_s = time.perf_counter() - t0
        steps = [a.step for a in parse_result.actions]

        t0 = time.perf_counter()
        timeline = scheduler.schedule(steps)
        timing.schedule_s = time.perf_counter() - t0
        for entry, action in zip(timeline, parse_result.actions):
            entry["label"] = action.raw.strip().rstrip(".")

        t0 = time.perf_counter()
        generator = VideoGenerator(clips_dir="clips", output_dir=os.path.join(output_dir, "processed"))
        generation = generator.generate(timeline)
        timing.generate_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        assembler = VideoAssembler(output_path=os.path.join(output_dir, f"{label}.mp4"))
        labels = [c.label for c in generation.clips]
        assembly = assembler.assemble(generation.clip_paths, transition=transition, labels=labels)
        timing.assemble_s = time.perf_counter() - t0

        timing.total_s = time.perf_counter() - t_start

        clip_sources: Dict[str, int] = {}
        for c in generation.clips:
            clip_sources[c.source] = clip_sources.get(c.source, 0) + 1

        return RunResult(
            label=label, instruction=instruction, timing=timing,
            num_actions=len(steps), steps=steps, clip_sources=clip_sources,
            final_duration_s=assembly.duration,
            parser_warning_count=len(parse_result.warnings),
            generation_warning_count=len(generation.warnings),
            status="PASS",
        )
    except Exception as exc:
        timing.total_s = time.perf_counter() - t_start
        return RunResult(
            label=label, instruction=instruction, timing=timing,
            num_actions=0, steps=[], clip_sources={}, final_duration_s=0.0,
            parser_warning_count=0, generation_warning_count=0,
            status="ERROR", error=str(exc),
        )


def main():
    output_dir = "benchmark_output"
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for label, instruction in BENCHMARK_INSTRUCTIONS:
        print(f"\n=== Running: {label} ===")
        result = run_once(label, instruction, output_dir)
        print(f"  status={result.status} total={result.timing.total_s:.2f}s actions={result.num_actions}")
        results.append(result)

    with open(os.path.join(output_dir, "benchmark_results.json"), "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nSaved {len(results)} results to {output_dir}/benchmark_results.json")


if __name__ == "__main__":
    main()
