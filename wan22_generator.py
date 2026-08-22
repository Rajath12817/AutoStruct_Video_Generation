"""
Wan 2.2 chained generation (Replicate) -- proof-of-concept Stage 4 alternative.

Wires into the exact `fallback_generator` extension point already built into
video_generator.py. First call has no prior frame, so it uses T2V
(wan-video/wan-2.2-t2v-fast); every call after that extracts the last frame of
the previously generated clip and uses I2V (wan-video/wan-2.2-i2v-fast) so
each new clip continues visually from where the last one ended.

Safety model, given this is a real-money, run-once test:
  - Steps that cost nothing (schema lookup, duration math, frame extraction)
    are built and verified before anything that calls a paid endpoint.
  - `dry_run()` prints exactly what would be requested -- prompts, durations,
    frame/fps choices, per-clip and total estimated cost -- with zero network
    calls to the paid endpoint.
  - `generate()` (the real, paid path) is intentionally NOT wired to retry on
    failure: one bad response aborts the whole run rather than silently
    spending more.

Requires REPLICATE_API_TOKEN in the environment (see .env.example).
"""

import logging
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import List, Optional, Tuple

from moviepy import VideoFileClip

logger = logging.getLogger("wan22_generator")

T2V_MODEL = "wan-video/wan-2.2-t2v-fast"
I2V_MODEL = "wan-video/wan-2.2-i2v-fast"

# From the user's Replicate pricing screenshot: "interpolate" variant @ 480p.
VARIANT = "interpolate"
RESOLUTION = "480p"
PRICE_PER_VIDEO_USD = 0.065

# Wan 2.2 fast models: num_frames in [81, 100], frames_per_second in [5, 24].
MIN_NUM_FRAMES = 81
MAX_NUM_FRAMES = 100
MIN_FPS = 5
MAX_FPS = 24
FIXED_NUM_FRAMES = 81  # cheapest setting; fps is solved to hit the target duration

# Fixed scene-anchor appended to every prompt, so the model always knows what
# kind of scene this is regardless of which action is being generated. A bare
# action ("turn on tap") gave the T2V model nothing to ground itself on and it
# hallucinated an unrelated scene in testing; this anchor fixes that, and
# keeps framing/lighting/setting language identical across every clip's
# prompt, which matters more here than usual since I2V chaining means every
# later clip inherits whatever scene the first clip established.
PROMPT_ANCHOR = (
    "Close-up shot of a person's hands at a bathroom sink, "
    "realistic indoor lighting, static camera angle, no text overlays."
)


def build_prompt(label: str) -> str:
    action = label.strip().rstrip(".")
    return f"{action}. {PROMPT_ANCHOR}"


EXPAND_PROMPT_SYSTEM = """You write prompts for a text/image-to-video generation model.

Given a short action description, expand it into ONE or TWO sentences that spell out
the physical sub-steps and the visible outcome, so the video model has something
concrete to render instead of an ambiguous label.

Rules:
1. Describe ONLY the given action -- never add steps that aren't implied by it.
2. Be mechanically specific: what does a hand/object DO, and what visibly CHANGES
   as a result (e.g. foam appearing, water flowing, a towel moving).
3. Do not mention camera, lighting, or setting -- that is added separately.
4. Output ONLY the expanded description. No quotes, no explanation, no markdown."""


def expand_generation_prompt(label: str, groq_api_key: str) -> str:
    """
    Turns a short parsed action (e.g. "Apply soap to your hands") into a
    mechanically explicit description via Groq, so ANY parsed action gets
    this treatment automatically -- not just a hand-picked few. Falls back to
    the raw label, unexpanded, if the LLM call fails, so a network hiccup
    degrades to the old (thinner) behavior rather than crashing the run.
    """
    from parser import GroqLLM
    llm = GroqLLM(api_key=groq_api_key)
    try:
        expanded = llm.call(EXPAND_PROMPT_SYSTEM, label, temperature=0.3)
        return expanded.strip().strip('"')
    except Exception as exc:
        logger.warning("[wan22] prompt expansion failed (%s) -- using raw label '%s'.", exc, label)
        return label


def solve_duration_params(target_duration_s: float) -> Tuple[int, int, float]:
    """
    Given a target clip duration, return (num_frames, fps, actual_duration_s)
    for a Wan 2.2 fast model call. num_frames is fixed at its minimum (81,
    the cheapest setting); fps is chosen to make num_frames/fps land as close
    to the target as the model's supported fps range [5, 24] allows.

    Achievable range at num_frames=81 is ~3.375s (fps=24) to 16.2s (fps=5).
    Targets below ~3.375s land at the floor and get trimmed afterward by the
    existing VideoGenerator._fit_duration logic; targets above 16.2s would
    need a second num_frames search, but the scheduler in this project caps
    durations at 8s, so that branch is intentionally not built out here.
    """
    if target_duration_s <= 0:
        raise ValueError(f"target_duration_s must be positive, got {target_duration_s}")

    ideal_fps = FIXED_NUM_FRAMES / target_duration_s
    fps = max(MIN_FPS, min(MAX_FPS, round(ideal_fps)))
    actual_duration = FIXED_NUM_FRAMES / fps
    return FIXED_NUM_FRAMES, fps, actual_duration


def extract_last_frame(video_path: str, out_path: str) -> str:
    """Save the final frame of `video_path` as an image at `out_path`, for
    feeding into the next clip's I2V call as its starting frame."""
    clip = VideoFileClip(video_path)
    try:
        if not clip.duration or clip.duration <= 0:
            raise ValueError(f"'{video_path}' has zero/invalid duration -- cannot extract last frame")
        # a hair before the true end avoids occasional decode issues exactly at EOF
        t = max(0.0, clip.duration - 0.05)
        frame = clip.get_frame(t)
    finally:
        clip.close()

    from PIL import Image
    Image.fromarray(frame).save(out_path)
    return out_path


@dataclass
class PlannedCall:
    index: int
    step: str
    prompt: str
    model: str
    target_duration: float
    num_frames: int
    fps: int
    actual_duration: float
    source_image: Optional[str]
    estimated_cost_usd: float


def plan_calls(timeline: List[dict]) -> List[PlannedCall]:
    """Pure planning pass -- no network calls, no cost. Used by both dry_run()
    and generate() so the plan they act on is always identical."""
    calls = []
    for i, entry in enumerate(timeline):
        step = entry["step"]
        prompt = entry.get("label") or step.replace("_", " ")
        duration = float(entry.get("duration", 3))
        num_frames, fps, actual_duration = solve_duration_params(duration)
        calls.append(PlannedCall(
            index=i, step=step, prompt=prompt,
            model=T2V_MODEL if i == 0 else I2V_MODEL,
            target_duration=duration, num_frames=num_frames, fps=fps,
            actual_duration=actual_duration,
            source_image=None if i == 0 else f"<last frame of clip {i-1}>",
            estimated_cost_usd=PRICE_PER_VIDEO_USD,
        ))
    return calls


def dry_run(timeline: List[dict]) -> None:
    """Print exactly what a real run would request and cost -- zero network calls."""
    calls = plan_calls(timeline)
    print(f"\n{'='*78}\nWAN 2.2 DRY RUN -- {len(calls)} clip(s), variant={VARIANT}, resolution={RESOLUTION}\n{'='*78}")
    for c in calls:
        print(f"\n[{c.index}] step={c.step!r}")
        print(f"    model:            {c.model}")
        print(f"    prompt:           {c.prompt!r}")
        print(f"    source image:     {c.source_image or '(none -- text-to-video)'}")
        print(f"    target duration:  {c.target_duration:.2f}s")
        print(f"    num_frames/fps:   {c.num_frames} / {c.fps}  ->  actual ~{c.actual_duration:.2f}s "
              f"(trimmed to {c.target_duration:.2f}s by the existing pipeline afterward)")
        print(f"    estimated cost:   ${c.estimated_cost_usd:.3f}")
    total = sum(c.estimated_cost_usd for c in calls)
    print(f"\n{'-'*78}\nESTIMATED TOTAL: ${total:.2f} for {len(calls)} clip(s)\n{'='*78}\n")


def _require_token() -> str:
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "REPLICATE_API_TOKEN is not set. Add it to .env (see .env.example) before continuing."
        )
    return token


def verify_schema() -> None:
    """
    STEP 1 -- free, metadata-only. Fetches the real input schema for both
    models from Replicate so the actual request-building code (WanChainedGenerator)
    is written against confirmed field names, not guessed ones. Costs nothing --
    this does not run a prediction.
    """
    _require_token()
    import replicate

    for model_id in (T2V_MODEL, I2V_MODEL):
        print(f"\n{'='*78}\nSCHEMA: {model_id}\n{'='*78}")
        model = replicate.models.get(model_id)
        version = model.latest_version
        schema = version.openapi_schema if version else None
        if not schema:
            print("  (no version/schema found -- model id may be wrong)")
            continue
        props = schema.get("components", {}).get("schemas", {}).get("Input", {}).get("properties", {})
        required = set(schema.get("components", {}).get("schemas", {}).get("Input", {}).get("required", []))
        for name, spec in props.items():
            mark = "*" if name in required else " "
            default = spec.get("default", "-")
            enum = spec.get("enum")
            enum_str = f" enum={enum}" if enum else ""
            print(f"  {mark} {name:<22} type={spec.get('type', spec.get('allOf', '?'))!s:<10} "
                  f"default={default!s:<12}{enum_str}")
        print("  (* = required)")


class Wan22GenerationError(Exception):
    """Raised on any failure in the real (paid) generation path. Never retried
    automatically -- a run-once, real-money test should stop loudly, not
    silently retry and spend more."""


class Wan22ChainedGenerator:
    """
    Real, paid generator. Call 0 uses T2V (no prior frame exists); every call
    after extracts the previous call's output's last frame and uses I2V, so
    each clip continues visually from where the last one ended.

    Matches the `fallback_generator: Callable[[str, int], str]` signature
    VideoGenerator expects, so it plugs directly into the pipeline via
    VideoGenerator(fallback_generator=Wan22ChainedGenerator(...)).
    """

    POLL_INTERVAL_S = 4
    MAX_WAIT_S = 300

    def __init__(self, output_dir: str = "wan22_output", groq_api_key: Optional[str] = None):
        _require_token()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._call_index = 0
        self._last_frame_path: Optional[str] = None
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")

    def resume_from(self, video_path: str, index: int) -> None:
        """
        Prime the chain from an already-generated clip instead of starting
        over -- lets a run-once budget be spent incrementally across separate
        script invocations without regenerating (and re-billing for) clips
        that already turned out correct.
        `index` is the position this clip occupied (0 = it was the T2V clip);
        the NEXT call will be index+1 and will use I2V seeded from its frame.
        """
        frame_path = os.path.join(self.output_dir, f"{index:02d}_resumed_last_frame.jpg")
        extract_last_frame(video_path, frame_path)
        self._last_frame_path = frame_path
        self._call_index = index + 1
        logger.info("[wan22] resumed from '%s' -- next call will be index %d (I2V)",
                    video_path, self._call_index)

    def __call__(self, step: str, duration: int, label: Optional[str] = None, expand: bool = True) -> str:
        import replicate

        index = self._call_index
        num_frames, fps, _actual_duration = solve_duration_params(float(duration))
        is_first = index == 0
        model_id = T2V_MODEL if is_first else I2V_MODEL

        raw_label = label or step.replace("_", " ")
        if expand and self.groq_api_key:
            action_desc = expand_generation_prompt(raw_label, self.groq_api_key)
        else:
            if expand:
                logger.warning("[wan22] no GROQ_API_KEY available -- using unexpanded label '%s'.", raw_label)
            action_desc = raw_label
        prompt = build_prompt(action_desc)

        model_input = dict(
            prompt=prompt,
            num_frames=num_frames,
            frames_per_second=fps,
            resolution=RESOLUTION,
            interpolate_output=True,  # matches the "interpolate" variant pricing
        )

        with ExitStack() as stack:
            if is_first:
                logger.info("[wan22] clip %d ('%s') -- T2V, no prior frame", index, step)
            else:
                logger.info("[wan22] clip %d ('%s') -- I2V, seeded from clip %d's last frame",
                            index, step, index - 1)
                model_input["image"] = stack.enter_context(open(self._last_frame_path, "rb"))
            logger.info("[wan22] prompt: %r", prompt)

            output_url = self._run_prediction(replicate, model_id, model_input, index, step)

        out_path = self._save_output(output_url, index, step)
        self._advance(out_path)
        return out_path

    def _run_prediction(self, replicate, model_id: str, model_input: dict, index: int, step: str) -> str:
        """
        Create the prediction (fast, negligible timeout risk) and poll for
        completion ourselves, rather than using replicate.run()'s built-in
        blocking wait. This matters specifically because a client-side read
        timeout on a one-shot call does NOT mean the (already billed)
        generation failed server-side -- polling an explicit prediction id
        means we never lose track of a job we've already paid for, and never
        risk creating a duplicate one by "retrying" a call that actually
        succeeded.
        """
        try:
            prediction = replicate.predictions.create(model=model_id, input=model_input)
        except Exception as exc:
            raise Wan22GenerationError(f"clip {index} ('{step}') {model_id}: failed to create prediction: {exc}") from exc

        logger.info("[wan22] prediction id=%s -- polling...", prediction.id)
        waited = 0
        while prediction.status not in ("succeeded", "failed", "canceled"):
            time.sleep(self.POLL_INTERVAL_S)
            waited += self.POLL_INTERVAL_S
            try:
                prediction.reload()
            except Exception as exc:
                raise Wan22GenerationError(
                    f"clip {index} ('{step}'): lost track of prediction {prediction.id} while polling "
                    f"({exc}). It may still complete/be billed -- check "
                    f"https://replicate.com/p/{prediction.id} before retrying."
                ) from exc
            if waited >= self.MAX_WAIT_S:
                raise Wan22GenerationError(
                    f"clip {index} ('{step}'): prediction {prediction.id} still '{prediction.status}' "
                    f"after {waited}s. It is NOT necessarily failed -- do not retry blindly. "
                    f"Check https://replicate.com/p/{prediction.id} and resume from it once it completes."
                )

        if prediction.status != "succeeded":
            raise Wan22GenerationError(
                f"clip {index} ('{step}') {model_id} prediction {prediction.id} ended in "
                f"status '{prediction.status}': {prediction.error}"
            )
        logger.info("[wan22] prediction %s succeeded after %ds", prediction.id, waited)
        return prediction.output

    def _save_output(self, output, index: int, step: str) -> str:
        video_path = os.path.join(self.output_dir, f"{index:02d}_{step}.mp4")
        try:
            if hasattr(output, "read"):
                with open(video_path, "wb") as f:
                    f.write(output.read())
            else:
                import urllib.request
                urllib.request.urlretrieve(str(output), video_path)
        except Exception as exc:
            raise Wan22GenerationError(f"clip {index} ('{step}'): failed to save model output: {exc}") from exc

        if not os.path.isfile(video_path) or os.path.getsize(video_path) == 0:
            raise Wan22GenerationError(f"clip {index} ('{step}'): downloaded file is missing or empty")
        return video_path

    def _advance(self, generated_video_path: str) -> None:
        frame_path = os.path.join(self.output_dir, f"{self._call_index:02d}_last_frame.jpg")
        extract_last_frame(generated_video_path, frame_path)
        self._last_frame_path = frame_path
        self._call_index += 1
