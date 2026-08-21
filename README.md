# Auto-Struct — Sequential Text-to-Video Pipeline (Handwashing)

## Project Overview

Converts a natural-language instruction paragraph into a single stitched
video where each described action appears, in order, for a duration that
matches how long that action actually takes.

```
Text instruction
   │
   ▼
Stage 1-2  Parser            parser.py           noisy text -> ordered atomic actions
   │
   ▼
Stage 3    Scheduler         scheduler.py        actions -> timeline (start/end/duration)
   │
   ▼
Stage 4    VideoGenerator    video_generator.py  timeline -> one trimmed/looped clip per action
   │
   ▼
Stage 5    VideoAssembler    assembler.py         ordered clips -> final_video.mp4
```

Run the whole thing end-to-end with:
```bash
python generate_video.py "turn on tap wet hands apply soap rinse dry"
```

**Current implementation status — Stage 4 is clip *retrieval*, not generation.**
`video_generator.py` maps each action to a pre-recorded clip in `clips/`
(same person, same static camera, same background, filmed by us) and
trims or loops it to the scheduler's assigned duration. No text-to-video
model (VideoCrafter, AnimateDiff, etc.) is currently in the loop — the
codebase has a `fallback_generator` hook in `VideoGenerator` for wiring
one in later, but nothing is connected yet. This was a deliberate choice:
our GPU (4GB VRAM) can't run those models at usable quality, and
retrieval trivially satisfies the "same person/background/camera"
consistency requirement that generative models struggle with. See
`claudePrompt.md` for the original problem statement, which described
Stage 4 in terms of a pretrained T2V model — that gap between spec and
implementation is intentional and should be called out as a scoping
decision in the capstone report, not treated as an oversight.

Planned next step: once retrieval-based output is validated against real
footage, evaluate free/open text-to-video models as a Stage 4 upgrade
(candidates: ModelScope T2V, AnimateDiff, CogVideoX, LTX-Video — via a
hosted API rather than local inference, given the VRAM constraint).

---

## Stage 1-2: Instruction Parser

A 6-layer NLP pipeline that converts noisy, informal, misspelled natural
language into clean structured action representations for video generation.

**Free API used:** [Groq](https://console.groq.com/) — llama-3.3-70b-versatile  
No credit card needed. Free tier: 14,400 requests/day.

---

## Architecture

```
Raw Input
    │
    ▼
┌─────────────────────────────────────────────┐
│ Layer 1 — Noise + Spelling Correction        │
│  · Domain vocab corrections (handwash words)│
│  · SymSpell compound correction (82k words) │
│  "wwash hand" → "wash hand"                 │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│ Layer 2 — LLM Grammar Normalization          │
│  · Fix tense: "takes" → "take"              │
│  · Fix grammar: "wash hand" → "wash hands"  │
│  · NO new steps added (strict prompt)       │
│  · Model: llama-3.3-70b via Groq (free)     │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│ Layer 3 — LLM Single-Action Splitting        │
│  · Splits compound sentences intelligently  │
│  · "wash hands and apply soap" → 2 actions  │
│  · "pick up and open bottle" → 1 action     │
│  · Output: JSON array of strings            │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│ Layer 4 — Verb + Object Extraction           │
│  · Rule-based heuristics (no NLTK needed)   │
│  · Imperative-aware: first non-stop = verb  │
│  · Domain noun list for object resolution   │
│  · Fallback: LLM extracts if low-confidence │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│ Layer 5 — Action Normalization               │
│  · Lemmatize verb: "washes" → "wash"        │
│  · Build snake_case: "wash" + "hands"       │
│                      → "wash_hands"         │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│ Layer 6 — Validation Guard                  │
│  · Detect duplicates (warns, never removes) │
│  · Detect low-confidence extractions        │
│  · Preserve original user-defined order     │
└────────────────────┬────────────────────────┘
                     │
                     ▼
Structured Action List (JSON)
[
  {"step": "turn_on_tap",   "verb": "turn", "object": "tap"},
  {"step": "wet_hands",     "verb": "wet",  "object": "hands"},
  {"step": "apply_soap",    "verb": "apply","object": "soap"},
  {"step": "lather_hands",  "verb": "lather","object":"hands"},
  {"step": "scrub_hands",   "verb": "scrub","object": "hands"},
  {"step": "rinse_hands",   "verb": "rinse","object": "hands"},
  {"step": "dry_hands",     "verb": "dry",  "object": "hands"}
]
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

`moviepy` (used by Stage 4/5) bundles its own static ffmpeg binary via
`imageio-ffmpeg` — no separate system ffmpeg install needed.

### 2. Get a free Groq API key

1. Go to [https://console.groq.com/](https://console.groq.com/)
2. Sign up (no credit card required)
3. Create an API key
4. Create a local `.env` file in the project root:

```bash
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

If you prefer the shell, exporting the variable also works:

```bash
export GROQ_API_KEY=gsk_your_key_here
export GROQ_MODEL=openai/gpt-oss-120b
```

### 3. Run

```python
from parser import InstructionParser

parser = InstructionParser()

# Single call
result = parser.parse("wwash hand and takes bottle near table")

# Get JSON output
json_output = parser.parse_to_json(
    "wet hands then aply soaap rub and rince off"
)
print(json_output)
```

### 4. Run tests

```bash
python test_parser.py
```

---

## Example I/O

### Input
```
"wwash hand and takes bottle near table"
```

### Layer-by-layer trace
```
[L1] Input      : 'wwash hand and takes bottle near table'
[L1] Cleaned    : 'wash hand and takes bottle near table'
[L2] Normalized : 'Wash your hands and take the bottle from the table.'
[L3] Split into : ['Wash your hands', 'Take the bottle from the table']
[L4] 'Wash your hands'             → verb='wash', obj='hands' → step='wash_hands'
[L4] 'Take the bottle from table'  → verb='take', obj='bottle' → step='take_bottle'
```

### Output JSON
```json
[
  {
    "step": "wash_hands",
    "verb": "wash",
    "object": "hands",
    "modifier": null,
    "raw": "Wash your hands"
  },
  {
    "step": "take_bottle",
    "verb": "take",
    "object": "bottle",
    "modifier": null,
    "raw": "Take the bottle from the table"
  }
]
```

---

## Design Constraints (Strictly Followed)

| Constraint | How enforced |
|---|---|
| No implicit step expansion | LLM system prompt explicitly forbids it |
| No hardcoded workflows | Domain vocab only for spelling, not for logic |
| Preserves user's order | Sentences processed sequentially, never reordered |
| Generalizable | Groq LLM handles any domain; only spelling dict is domain-tuned |
| Robust to noise | SymSpell + domain corrections handle the worst typos |

---

## Files

```
sentence_parser/
├── parser.py         — full 6-layer pipeline
├── test_parser.py    — comprehensive test suite (12 test cases)
└── README.md         — this file
```

---

## Stage 3: Temporal Scheduler

The parser's output feeds directly into `scheduler.py`, which assigns a
realistic duration to each action (LLM-estimated, grounded by statistical
averages computed from `handwash_dataset.csv`) and lays them out as a
non-overlapping timeline:

```python
actions = [a.step for a in result.actions]
# → ["turn_on_tap", "wet_hands", "apply_soap", "rinse_hands", "dry_hands"]

timeline = scheduler.schedule(actions)
# → [
#   {"step": "turn_on_tap", "start": 0, "end": 2, "duration": 2},
#   {"step": "wet_hands",   "start": 2, "end": 5, "duration": 3},
#   ...
# ]
```

## Stage 4-5: Clip Retrieval + Assembly

`video_generator.py` turns that timeline into one normalized clip per
action (trimmed/looped from `clips/<step_name>.mp4`), and `assembler.py`
concatenates them in order into `output/final_video.mp4`. See the
project overview at the top of this file for the current status and
limitations of this stage.

## Clip Bank Convention

Place one file per action in `clips/`, named after the parser's `step`
value, all filmed with the same person/background/static camera:

```
clips/turn_on_tap.mp4
clips/wet_hands.mp4
clips/apply_soap.mp4
clips/lather_hands.mp4
clips/scrub_hands.mp4
clips/rinse_hands.mp4
clips/dry_hands.mp4
```

If a step has no matching clip, `VideoGenerator` falls back to a plain
colored placeholder clip rather than failing — useful for smoke-testing
the pipeline, not for a final demo.
