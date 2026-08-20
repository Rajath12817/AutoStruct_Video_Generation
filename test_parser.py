"""
Test Suite — Advanced Instruction-to-Video Sentence Parser
==========================================================
Tests cover:
  - Noisy / misspelled handwash instructions
  - Multi-action sentences (and / then / comma-separated)
  - Informal language
  - Mixed tense and wrong verb forms
  - General action instructions
  - Edge cases (single word, empty-ish, repeated steps)

Set your Groq API key before running:
    export GROQ_API_KEY=your_actual_key
    python test_parser.py
"""

import os
import json
from parser import InstructionParser

# ── Setup ─────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError("Set GROQ_API_KEY before running the parser tests.")

parser = InstructionParser(groq_api_key=API_KEY)
DIVIDER = "─" * 65

def run_test(label: str, prompt: str):
    print(f"\n{DIVIDER}")
    print(f"TEST: {label}")
    print(f"INPUT: {prompt!r}")
    print(DIVIDER)
    result = parser.parse(prompt, verbose=True)
    print(f"\n{'─'*30} FINAL OUTPUT {'─'*30}")
    for i, a in enumerate(result.actions, 1):
        print(f"  Step {i:02d}: {a.step:<30}  (verb={a.verb}, obj={a.obj})")
    if result.warnings:
        print(f"\n  Warnings: {result.warnings}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TEST CASES
# ══════════════════════════════════════════════════════════════════════════════

tests = [
    # ── Handwash: noisy / misspelled ──────────────────────────────────────────
    (
        "Handwash - heavily misspelled",
        "wwash hand and takes bottle near table",
    ),
    (
        "Handwash - WHO 7-step (informal, no punctuation)",
        "wet hands with water then aply soaap rub palms together "
        "interlok fingers scrub backs of hands rubb thumbs rince with "
        "water and drie with towel",
    ),
    (
        "Handwash - mixed tense + typos",
        "first trun on the tap then wets your hand aply some soaap "
        "then she scrubs and rinced off finally dry hand",
    ),
    (
        "Handwash - very informal slang",
        "yo open tap get hands wet put soap on hands rub rub rub "
        "get the soap off then dry it",
    ),
    (
        "Handwash - single run-on sentence",
        "turn on tap wet hands apply soap lather scrub rinse dry",
    ),
    (
        "Handwash - numbered list style",
        "1 turn on tap 2 wet your hands 3 apply soap 4 rub hands "
        "5 rinse off soap 6 turn off tap 7 dry hands with towel",
    ),
    (
        "Handwash - clean well-formed input",
        "Turn on the tap. Wet your hands. Apply soap. Scrub your "
        "hands for 20 seconds. Rinse your hands. Dry with a clean towel.",
    ),

    # ── General actions: noisy ────────────────────────────────────────────────
    (
        "General - pour and stir (compound)",
        "pour water in the pot and stirr it",
    ),
    (
        "General - complex multi-step cooking",
        "chop onins then fry on oile add salt and peper pour watter "
        "and biol for 10 minuts",
    ),
    (
        "General - making tea",
        "boil water put tea bag in cup pour water wait 3 mins remove tea bag add sugar",
    ),
    (
        "General - changing bulb",
        "turn off light switch get chair stand on chair unscrew old bulb screw in new bulb get down",
    ),
    (
        "General - exercise routine",
        "get down on floor do 10 push ups stand back up",
    ),
    (
        "General - single action word",
        "wash",
    ),

    # ── Edge cases ────────────────────────────────────────────────────────────
    (
        "Edge - duplicate steps (non-consecutive)",
        "wash hands then wash hands again then dry hands",
    ),
    (
        "Edge - consecutive duplicate verbs",
        "scrub scrub scrub hands then rinse",
    ),
    (
        "Edge - action + location noise",
        "takes bottle near table and opens it",
    ),
    (
        "Edge - tea bag sequence",
        "Boil water, place tea bag in cup, pour hot water, wait 3 minutes, remove tea bag, add sugar, stir",
    ),
    (
        "Edge - action + location noise",
        "enter bathroom turn tap wash hands apply soap rinse hands",
    ),
]

# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*65)
print("  INSTRUCTION PARSER — FULL TEST RUN")
print("═"*65)

results_summary = []

for label, prompt in tests:
    try:
        result = run_test(label, prompt)
        results_summary.append({
            "test": label,
            "input": prompt,
            "steps": [a.step for a in result.actions],
            "warnings": result.warnings,
            "status": "PASS",
        })
    except Exception as e:
        print(f"\n[ERROR] Test '{label}' failed: {e}")
        results_summary.append({
            "test": label,
            "input": prompt,
            "steps": [],
            "warnings": [],
            "status": f"ERROR: {e}",
        })

# ── Summary report ────────────────────────────────────────────────────────────
print(f"\n\n{'═'*65}")
print("  SUMMARY REPORT")
print("═"*65)

for r in results_summary:
    status = "✓" if r["status"] == "PASS" else "✗"
    print(f"\n{status} {r['test']}")
    print(f"  Input  : {r['input'][:70]}{'...' if len(r['input'])>70 else ''}")
    print(f"  Steps  : {r['steps']}")
    if r["warnings"]:
        print(f"  Warns  : {r['warnings']}")
    print(f"  Status : {r['status']}")

# ── Save full JSON output ─────────────────────────────────────────────────────
with open("test_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"\n\nFull results saved to test_results.json")
print("Done.")
