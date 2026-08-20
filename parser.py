"""
Advanced Instruction-to-Video Sentence Parser
==============================================
Pipeline: Noise Correction → LLM Normalization → LLM Splitting
          → Dependency Extraction → Action Normalization → Validation

Free API used: Groq (llama-3.3-70b-versatile) — free tier, no credit card needed
Sign up: https://console.groq.com/
"""

import re
import json
import os
import time
from typing import Optional
from dataclasses import dataclass, asdict

from symspellpy import SymSpell, Verbosity
from groq import Groq
import spacy
import nltk
from nltk.stem import WordNetLemmatizer


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.3-70b-versatile"   # free & powerful

# ─────────────────────────────────────────────
#  DOMAIN VOCABULARY  (handwash-focused + general)
#  These override SymSpell when domain words
#  would be corrected to wrong common words
# ─────────────────────────────────────────────

DOMAIN_CORRECTIONS = {
    # misspelled → correct
    "wwash": "wash",  "wosh": "wash",   "wassh": "wash",   "wsh": "wash",
    "rince": "rinse", "rins": "rinse",  "rnse": "rinse",   "rinsse": "rinse",
    "aply":  "apply", "appli": "apply", "appley": "apply",
    "soaap": "soap",  "sopa": "soap",   "soaps": "soap",
    "lather":"lather","lathe": "lather","lathre": "lather",
    "rubb":  "rub",   "ruub": "rub",    "scrub": "scrub",  "srubb": "scrub",
    "drie":  "dry",   "drry": "dry",
    "trun":  "turn",  "trn": "turn",
    "wett":  "wet",   "wett": "wet",
    "taek":  "take",  "taeks": "takes",
    "dispence": "dispense", "disspense": "dispense",
    "interlok": "interlock", "intelace": "interlace",
    "betwen":  "between",   "fingres": "fingers",
    "thuumbs": "thumbs",    "thums": "thumbs",
    "nails":   "nails",     "plam": "palm",
    "writs":   "wrists",    "wrist": "wrist",
    "sanitizer": "sanitizer", "santizer": "sanitizer",
    # slang openers / filler words that SymSpell maps to wrong words
    "yo": "",    "gonna": "going to", "gotta": "got to",
    "lemme": "let me", "gimme": "give me",
    # double-error words SymSpell struggles with
    "rinceed": "rinse", "rinseed": "rinse", "rinsed": "rinse",
    "scrubbs": "scrub", "rubing": "rub", "rubing": "rub",
    "appply": "apply",  "applay": "apply",
    "wetts": "wet",     "dryed": "dry",
    # cooking / general domain
    "onins": "onions", "onons": "onions", "oinons": "onions",
    "oile": "oil",     "oyl": "oil",
    "peper": "pepper", "peppr": "pepper",
    "watter": "water", "watr": "water",
    "biol": "boil",    "boill": "boil",
    "stirr": "stir",   "ster": "stir",
    "chiken": "chicken","letuce": "lettuce",
}

# Irregular + domain verb → base lemma



# ─────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class ParsedAction:
    step:     str            # e.g. "wash_hands"
    verb:     str            # e.g. "wash"
    obj:      str            # e.g. "hands"
    modifier: Optional[str]  # e.g. "thoroughly" or None
    raw:      str            # original sentence before normalization

@dataclass
class ParserResult:
    actions:       list[ParsedAction]
    cleaned_input: str        # after noise correction
    normalized:    str        # after LLM grammar fix
    split_sentences: list[str]
    warnings:      list[str]


# ─────────────────────────────────────────────
#  LAYER 1 — NOISE + SPELLING CORRECTION
# ─────────────────────────────────────────────

class SpellingCorrector:
    """
    Uses SymSpell (edit-distance dictionary lookup) with bundled
    82k-word English frequency dictionary + bigram dictionary.
    Domain corrections applied first to prevent wrong fixups
    (e.g. 'rubb' → 'ruby' by SymSpell, but we want 'rub').
    """

    def __init__(self):
        import pkg_resources, os as _os
        sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        pkg_dir = _os.path.dirname(__import__("symspellpy").__file__)
        sym.load_dictionary(
            _os.path.join(pkg_dir, "frequency_dictionary_en_82_765.txt"),
            term_index=0, count_index=1
        )
        sym.load_bigram_dictionary(
            _os.path.join(pkg_dir, "frequency_bigramdictionary_en_243_342.txt"),
            term_index=0, count_index=2
        )
        self.sym = sym

    def _apply_domain_corrections(self, text: str) -> str:
        """Token-level domain dictionary applied before/after SymSpell."""
        tokens = text.lower().split()
        corrected = []
        for tok in tokens:
            # strip trailing punctuation for lookup, re-attach after
            clean  = tok.strip(".,!?;:'\"")
            suffix = tok[len(clean):]
            replacement = DOMAIN_CORRECTIONS.get(clean, clean)
            if replacement == "":
                continue          # drop slang filler (e.g. "yo")
            corrected.append(replacement + suffix)
        return " ".join(corrected)

    def correct(self, text: str) -> str:
        # Step 0: strip numbered list markers (e.g. "1 wash 2 rinse" → "wash rinse")
        text = re.sub(r'\b\d+[\.\):]?\s*', ' ', text).strip()
        # Step 1: lowercase + domain fixes (catches obvious misspellings first)
        text = self._apply_domain_corrections(text)
        if not text.strip():
            return text
        # Step 2: SymSpell compound correction (handles remaining multi-word errors)
        suggestions = self.sym.lookup_compound(text, max_edit_distance=2)
        corrected = suggestions[0].term if suggestions else text
        # Step 3: re-apply domain corrections (SymSpell may undo domain fixes)
        corrected = self._apply_domain_corrections(corrected)
        # Step 4: clean up any double spaces from token removal
        corrected = " ".join(corrected.split())
        return corrected


# ─────────────────────────────────────────────
#  GROQ LLM CLIENT  (free API)
# ─────────────────────────────────────────────

class GroqLLM:
    """
    Thin wrapper around Groq's free API.
    Model: llama-3.3-70b-versatile
    Sign up free at: https://console.groq.com/
    """

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model  = GROQ_MODEL

    def call(self, system: str, user: str, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
#  LAYER 2 — LLM GRAMMAR NORMALIZATION
# ─────────────────────────────────────────────

NORMALIZE_SYSTEM = """You are a grammar normalizer for an instruction parser.

Rules:
1. Rewrite the input as grammatically correct, present-tense imperative English.
2. Fix grammar and tense. Do NOT add or remove any actions.
3. Resolve any pronouns (e.g., 'it', 'them') or missing implied objects to their explicit nouns based on context. 
   Example: "chop onions and fry" → "Chop the onions and fry the onions."
   Example: "apply soap and scrub" → "Apply soap and scrub your hands."
4. Do NOT expand implicit steps (e.g. "wash hands" stays as-is, never expand it).
5. Keep the original meaning exactly. Just fix the grammar and tense.
6. Output only the corrected text. No explanation, no quotation marks."""

def normalize_grammar(llm: GroqLLM, text: str) -> str:
    result = llm.call(NORMALIZE_SYSTEM, text)
    # Safety: strip any wrapping quotes the model adds
    return result.strip('"\'')


# ─────────────────────────────────────────────
#  LAYER 3 — LLM SINGLE-ACTION SPLITTING
# ─────────────────────────────────────────────

SPLIT_SYSTEM = """You are a sentence splitter for an instruction-to-video system.

Rules:
1. Split the input into individual sentences, one per physical action.
2. NEVER invent, add, or assume actions not present in the input.
3. Do NOT expand compound verbs unless they are clearly two distinct physical actions.
   Example: "pick up and open the bottle" → keep as ONE action (it is one motion).
   Example: "wash your hands and apply soap" → split into TWO actions.
4. Preserve the original order.
5. Return ONLY a JSON array of strings. No explanation, no markdown fences.
   Example output: ["turn on the tap", "wet your hands", "apply soap"]"""

def split_into_actions(llm: GroqLLM, text: str) -> list[str]:
    raw = llm.call(SPLIT_SYSTEM, text)
    # Strip markdown fences if model adds them
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        sentences = json.loads(raw)
        if isinstance(sentences, list):
            return [str(s).strip() for s in sentences if str(s).strip()]
    except json.JSONDecodeError:
        pass
    # Fallback: split by newlines / bullet markers
    lines = re.split(r"[\n;]+", raw)
    cleaned = []
    for line in lines:
        line = re.sub(r"^[\d\-\.\*\)\s]+", "", line).strip()
        if line:
            cleaned.append(line)
    return cleaned


# ─────────────────────────────────────────────
#  LAYER 4 — DEPENDENCY / STRUCTURE EXTRACTION
#  (Uses spaCy en_core_web_sm)
# ─────────────────────────────────────────────

def extract_verb_object_spacy(nlp, sentence: str) -> tuple[str, str, Optional[str]]:
    """
    Extracts (verb, object, modifier) from an imperative sentence using spaCy.
    Falls back to LLM extraction (Layer 4b) if heuristics fail.
    """
    doc = nlp(sentence)
    
    verb = None
    obj = None
    modifier = None
    
    # 1. Find the root verb
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            verb = token.text
            
            dobj_node = None
            prep_nodes = []
            
            for child in token.children:
                if child.dep_ == "dobj":
                    dobj_node = child
                elif child.dep_ == "prep":
                    prep_nodes.append(child)
                elif child.dep_ == "pobj" and not dobj_node:
                    dobj_node = child
                elif child.dep_ == "advmod":
                    modifier = child.text
            
            obj_parts = []
            if dobj_node:
                obj_parts.append(" ".join([t.text for t in dobj_node.subtree]))
            for prep in prep_nodes:
                obj_parts.append(" ".join([t.text for t in prep.subtree]))
                
            if obj_parts:
                obj = " ".join(obj_parts)
            break
            
    # If no ROOT verb found, fallback to first verb
    if not verb:
        for token in doc:
            if token.pos_ == "VERB":
                verb = token.text
                dobj_node = None
                prep_nodes = []
                for child in token.children:
                    if child.dep_ == "dobj":
                        dobj_node = child
                    elif child.dep_ == "prep":
                        prep_nodes.append(child)
                    elif child.dep_ == "pobj" and not dobj_node:
                        dobj_node = child
                    elif child.dep_ == "advmod":
                        modifier = child.text
                
                obj_parts = []
                if dobj_node:
                    obj_parts.append(" ".join([t.text for t in dobj_node.subtree]))
                for prep in prep_nodes:
                    obj_parts.append(" ".join([t.text for t in prep.subtree]))
                    
                if obj_parts:
                    obj = " ".join(obj_parts)
                break
                
    # If still no verb found, try to assume the first token is a verb (imperative structure)
    if not verb and len(doc) > 0:
        verb = doc[0].text
        for token in doc:
            if token.dep_ in ("dobj", "pobj"):
                obj = " ".join([t.text for t in token.subtree])
                break

    verb = verb if verb else "unknown"
    obj = obj if obj else "none"
        
    return verb, obj, modifier


# ─────────────────────────────────────────────
#  LAYER 4b — LLM-ASSISTED EXTRACTION
#  Used when heuristics produce low-confidence results
# ─────────────────────────="────────────────────

EXTRACT_SYSTEM = """You extract verb and object from an imperative sentence.
Include any prepositional phrases (locations, instruments, etc) in the object.

Output ONLY a JSON object with keys "verb" (base form) and "object" (full noun phrase + preposition).
No explanation, no markdown fences.
Example: {"verb": "place", "object": "tea bag in cup"}
Example: {"verb": "pour", "object": "hot water over tea bag"}"""

def llm_extract_verb_object(llm: GroqLLM, sentence: str) -> tuple[str, str]:
    raw = llm.call(EXTRACT_SYSTEM, sentence)
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        d = json.loads(raw)
        verb = str(d.get("verb","unknown")).lower().strip()
        obj  = str(d.get("object","none")).lower().strip().replace(" ","_")
        return verb, obj
    except Exception:
        return "unknown", "none"

def _confidence_ok(verb: str, obj: str) -> bool:
    """Returns False if extraction result looks unreliable."""
    return verb not in ("unknown","") and obj not in ("none","","unknown")


# ─────────────────────────────────────────────
#  LAYER 5 — ACTION NORMALIZATION
# ─────────────────────────────────────────────

def to_snake_case(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(a|an|the)\b", "", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("_")
    return text

def build_step_name(lemmatizer, verb: str, obj: str) -> str:
    """Combine lemmatized verb + object into canonical snake_case step name."""
    if verb not in ("unknown", "none", ""):
        verb = lemmatizer.lemmatize(verb.lower(), pos='v')
        
    verb_lem = to_snake_case(verb)
    obj_clean  = to_snake_case(obj)
    
    if obj_clean in ("none","","unknown","it","them"):
        return verb_lem
    # Avoid redundant combinations like wash_wash
    if verb_lem in obj_clean:
        return obj_clean
    return f"{verb_lem}_{obj_clean}"


# ─────────────────────────────────────────────
#  LAYER 6 — VALIDATION GUARD
# ─────────────────────────────────────────────

def validate_actions(actions: list[ParsedAction]) -> list[str]:
    """
    Returns a list of warning strings.
    Never removes steps — only warns.
    """
    warnings = []
    seen_steps = {}

    for i, action in enumerate(actions):
        # Duplicate check
        if action.step in seen_steps:
            warnings.append(
                f"Duplicate step '{action.step}' at position {i+1} "
                f"(first seen at {seen_steps[action.step]+1})"
            )
        seen_steps[action.step] = i

        # Unknown extraction
        if action.verb == "unknown" or action.obj == "none":
            warnings.append(
                f"Low-confidence extraction at step {i+1}: '{action.raw}' "
                f"→ verb={action.verb}, obj={action.obj}"
            )

    return warnings


# ─────────────────────────────────────────────
#  MAIN PIPELINE CLASS
# ─────────────────────────────────────────────

class InstructionParser:
    """
    Full 6-layer instruction parser pipeline.

    Usage:
        parser = InstructionParser(groq_api_key="gsk_...")
        result = parser.parse("wwash hand and takes bottle near table")
        print(result)
    """

    def __init__(self, groq_api_key: Optional[str] = None):
        print("[Parser] Initializing SymSpell corrector...")
        self.corrector = SpellingCorrector()
        print("[Parser] Loading spaCy dependency parser (en_core_web_sm)...")
        self.nlp = spacy.load("en_core_web_sm")
        print("[Parser] Loading NLTK Lemmatizer...")
        self.lemmatizer = WordNetLemmatizer()
        print("[Parser] Connecting to Groq LLM (free tier)...")
        api_key = groq_api_key or GROQ_API_KEY
        if not api_key:
            raise ValueError("Set GROQ_API_KEY in your environment before running the parser.")
        self.llm = GroqLLM(api_key=api_key)
        print("[Parser] Ready.\n")

    def parse(self, raw_input: str, verbose: bool = True) -> ParserResult:
        warnings = []

        # ── Layer 1: Noise + Spelling ──────────────────
        if verbose:
            print(f"[L1] Input      : {raw_input!r}")
        cleaned = self.corrector.correct(raw_input)
        if verbose:
            print(f"[L1] Cleaned    : {cleaned!r}")

        # ── Layer 2: LLM Grammar Normalization ─────────
        normalized = normalize_grammar(self.llm, cleaned)
        if verbose:
            print(f"[L2] Normalized : {normalized!r}")

        # ── Layer 3: LLM Single-Action Splitting ───────
        sentences = split_into_actions(self.llm, normalized)
        if verbose:
            print(f"[L3] Split into : {sentences}")

        # ── Layer 4+5: Extract + Normalize per sentence ─
        actions: list[ParsedAction] = []
        last_step = None
        for sent in sentences:
            verb, obj, modifier = extract_verb_object_spacy(self.nlp, sent)

            # Layer 4b: fallback to LLM if heuristics give low confidence
            if not _confidence_ok(verb, obj):
                if verbose:
                    print(f"[L4b] spacy fallback for '{sent}', using LLM...")
                verb, obj = llm_extract_verb_object(self.llm, sent)
                modifier = None

            step = build_step_name(self.lemmatizer, verb, obj)

            # Ignore consecutive identical steps (e.g., "rub rub rub")
            if step == last_step:
                continue
            last_step = step

            action = ParsedAction(
                step=step,
                verb=verb,
                obj=obj,
                modifier=modifier,
                raw=sent,
            )
            actions.append(action)
            if verbose:
                print(f"[L4] '{sent}' → verb={verb!r}, obj={obj!r} → step={step!r}")

        # ── Layer 6: Validation ─────────────────────────
        warnings = validate_actions(actions)
        if warnings and verbose:
            for w in warnings:
                print(f"[L6] WARNING: {w}")

        return ParserResult(
            actions=actions,
            cleaned_input=cleaned,
            normalized=normalized,
            split_sentences=sentences,
            warnings=warnings,
        )

    def parse_to_json(self, raw_input: str) -> str:
        """Returns final action list as JSON string."""
        result = self.parse(raw_input, verbose=False)
        output = [
            {
                "step":     a.step,
                "verb":     a.verb,
                "object":   a.obj,
                "modifier": a.modifier,
                "raw":      a.raw,
            }
            for a in result.actions
        ]
        return json.dumps(output, indent=2)
