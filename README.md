# Advanced Instruction-to-Video Sentence Parser

## Overview

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
pip install symspellpy groq
```

> No spaCy model download needed — uses bundled SymSpell dictionaries
> and Groq's hosted LLM API.

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

## Next Module: Temporal Scheduling (Stage 3)

The output of this parser feeds directly into the temporal scheduler:

```python
actions = [a.step for a in result.actions]
# → ["turn_on_tap", "wet_hands", "apply_soap", "rinse_hands", "dry_hands"]

# Temporal scheduler assigns start/end times per action:
# [
#   {"step": "turn_on_tap", "start": 0,  "end": 2},
#   {"step": "wet_hands",   "start": 2,  "end": 5},
#   ...
# ]
```
