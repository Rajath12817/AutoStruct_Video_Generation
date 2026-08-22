"""
Web UI for the full pipeline with Wan 2.2 as the Stage 4 generator.

One instruction goes in. The EXISTING parser and scheduler decide how many
actions there are and how long each one gets -- exactly as they do for the
retrieval pipeline in generate_video.py. Wan 2.2 only replaces Stage 4 (where
a clip comes from); it does not decide step count or duration. Nothing in
this UI lets the user override that -- there is no manual clip list, no
duration slider. The parser/scheduler's output is the plan, shown to the
user for review, and generation follows it exactly.

Flow: instruction -> Parse & Preview (parser + scheduler, free) -> review the
resulting timeline and its cost -> explicit confirmation -> Generate (paid,
Wan 2.2 chained on last-frame per clip, same hardened poll-based generator
used in CLI testing) -> assembled final video, preview + download.

Run with: streamlit run app.py
"""

import os

import streamlit as st
from moviepy import VideoFileClip

from env_loader import load_dotenv_file

load_dotenv_file()

from assembler import AssemblerError, VideoAssembler
from video_utils import normalize_clip
from wan22_generator import (
    PRICE_PER_VIDEO_USD,
    Wan22ChainedGenerator,
    Wan22GenerationError,
)

OUTPUT_DIR = "wan22_output"
FINAL_DIR = "output"
TARGET_RESOLUTION = (832, 480)  # matches RESOLUTION="480p" used by the Wan models

st.set_page_config(page_title="Auto-Struct -- Wan 2.2 Pipeline", page_icon="🎬", layout="centered")

st.title("🎬 Auto-Struct — Instruction to Video (Wan 2.2)")
st.caption(
    "Parser -> Scheduler -> Wan 2.2 generation -> Assembly. The instruction you type is "
    "decomposed into actions and timed automatically -- you don't choose clip count or "
    "duration, the pipeline does."
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set in .env -- required for the parser and scheduler.")
    st.stop()
if not os.environ.get("REPLICATE_API_TOKEN"):
    st.error("REPLICATE_API_TOKEN is not set in .env. Add it before using this page.")
    st.stop()


@st.cache_resource
def _load_parser():
    from parser import InstructionParser
    return InstructionParser(groq_api_key=GROQ_API_KEY)


@st.cache_resource
def _load_scheduler():
    from scheduler import TemporalScheduler
    return TemporalScheduler(groq_api_key=GROQ_API_KEY)


for key, default in [
    ("timeline", None), ("parser_warnings", []), ("session_spend", 0.0), ("result_path", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.divider()
st.subheader("1. Instruction")

instruction = st.text_area(
    "What should the video show?",
    placeholder='e.g. "Apply soap to your hands" or a full sequence like '
                '"Turn on the tap, wet your hands, apply soap, rinse, and dry them."',
    height=90,
)

if st.button("🔍 Parse & Preview (free -- no video generated yet)"):
    if not instruction.strip():
        st.warning("Enter an instruction first.")
    else:
        with st.spinner("Parsing instruction and scheduling durations..."):
            try:
                parser = _load_parser()
                scheduler = _load_scheduler()
                parse_result = parser.parse(instruction, verbose=False)
                steps = [a.step for a in parse_result.actions]
                if not steps:
                    st.session_state.timeline = None
                    st.error("The parser found no actions in that instruction -- try rephrasing.")
                else:
                    timeline = scheduler.schedule(steps)
                    for entry, action in zip(timeline, parse_result.actions):
                        entry["label"] = action.raw.strip().rstrip(".")
                    st.session_state.timeline = timeline
                    st.session_state.parser_warnings = parse_result.warnings
                    st.session_state.result_path = None
            except Exception as exc:
                st.session_state.timeline = None
                st.error(f"Parsing/scheduling failed: {exc}")

st.divider()
st.subheader("2. Plan (decided by the parser + scheduler, not editable here)")

timeline = st.session_state.timeline

if not timeline:
    st.caption("Parse an instruction above to see the plan.")
else:
    if st.session_state.parser_warnings:
        st.warning("Parser warnings: " + "; ".join(st.session_state.parser_warnings))

    st.table([
        {
            "#": i + 1, "Step": e["step"], "Action": e["label"],
            "Start": f"{e['start']}s", "End": f"{e['end']}s", "Duration": f"{e['duration']}s",
        }
        for i, e in enumerate(timeline)
    ])

    n_clips = len(timeline)
    estimated_cost = n_clips * PRICE_PER_VIDEO_USD
    total_duration = sum(e["duration"] for e in timeline)

    st.metric(
        "Estimated cost for this run", f"${estimated_cost:.3f}",
        help=f"${PRICE_PER_VIDEO_USD} per clip x {n_clips} clip(s), {total_duration}s final video",
    )

    expand_prompts = st.checkbox(
        "Auto-expand each action into a more descriptive prompt (via Groq)", value=True,
        help="Helps ground vague prompts (recommended). Does not fix motion-continuation "
             "bias in chained clips -- see project notes for that known limitation.",
    )
    transition = st.radio(
        "Transition between clips (only matters if 2+ actions)",
        options=["cut", "crossfade"], horizontal=True,
    )

    confirmed = st.checkbox(
        f"I understand this will spend approximately ${estimated_cost:.3f} on real API calls "
        f"to generate {n_clips} clip(s).",
        value=False,
    )

    generate_clicked = st.button("🚀 Generate", type="primary", disabled=not confirmed)

    st.divider()
    st.subheader("3. Result")

    def _trim_and_normalize(raw_path: str, duration: float, out_path: str) -> str:
        clip = VideoFileClip(raw_path)
        try:
            trimmed = clip.subclipped(0, min(duration, clip.duration))
            trimmed = normalize_clip(trimmed, target_resolution=TARGET_RESOLUTION, target_fps=24)
            trimmed.write_videofile(out_path, fps=24, codec="libx264", audio=False, logger=None)
            trimmed.close()
        finally:
            clip.close()
        return out_path

    if generate_clicked:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(FINAL_DIR, exist_ok=True)
        st.session_state.result_path = None

        try:
            gen = Wan22ChainedGenerator(output_dir=OUTPUT_DIR)
            trimmed_paths = []
            labels = []

            progress = st.empty()
            for i, entry in enumerate(timeline):
                progress.info(f"Generating clip {i + 1}/{n_clips}: “{entry['label']}” ...", icon="⏳")
                with st.spinner(f"Waiting on Wan 2.2 (clip {i + 1}/{n_clips}) -- this can take 30-100s..."):
                    raw_path = gen(
                        step=entry["step"], duration=entry["duration"],
                        label=entry["label"], expand=expand_prompts,
                    )
                trimmed_path = os.path.join(OUTPUT_DIR, f"ui_{i:02d}_trimmed.mp4")
                _trim_and_normalize(raw_path, entry["duration"], trimmed_path)
                trimmed_paths.append(trimmed_path)
                labels.append(entry["label"])
                st.session_state.session_spend += PRICE_PER_VIDEO_USD

            progress.empty()

            if n_clips == 1:
                final_path = os.path.join(FINAL_DIR, "ui_generated.mp4")
                os.replace(trimmed_paths[0], final_path)
            else:
                assembler = VideoAssembler(
                    output_path=os.path.join(FINAL_DIR, "ui_generated.mp4"),
                    target_resolution=TARGET_RESOLUTION,
                )
                result = assembler.assemble(trimmed_paths, transition=transition, labels=labels)
                final_path = result.output_path

            st.session_state.result_path = final_path
            st.success(f"Done -- spent ${n_clips * PRICE_PER_VIDEO_USD:.3f} on this run.")

        except (Wan22GenerationError, AssemblerError) as exc:
            st.error(f"Generation failed: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

    if st.session_state.result_path and os.path.isfile(st.session_state.result_path):
        st.video(st.session_state.result_path)
        with open(st.session_state.result_path, "rb") as f:
            st.download_button(
                "⬇️ Download video", data=f.read(),
                file_name=os.path.basename(st.session_state.result_path), mime="video/mp4",
            )

st.divider()
st.caption(f"Session spend so far (this app only): **${st.session_state.session_spend:.3f}**")
