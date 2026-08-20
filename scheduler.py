import csv
import json
import re
from typing import List, Dict, Any
from collections import defaultdict
from parser import GroqLLM

# The LLM System Prompt
SCHEDULER_SYSTEM = """You are an expert temporal scheduler for a text-to-video generation pipeline.

You will be given a list of sequential physical actions.
Your task is to assign a realistic duration (in seconds) to each action.

Below are the statistical average durations (in seconds) for some known actions based on our dataset:
{dataset_averages}

Rules:
1. If an input action is semantically similar to a known action above, use a duration close to its statistical average.
2. If an input action is UNKNOWN, intelligently estimate a realistic duration (e.g. chop_onions might take 5-8s).
3. Return ONLY a valid JSON array of objects, where each object has:
   - "action": the exact action name provided.
   - "duration": the estimated duration in seconds (MUST be a whole number integer).
4. Do NOT output any markdown blocks, fences, or explanations. ONLY the JSON array.

Example input:
["turn_on_tap", "wet_hands", "rub_hands"]

Example output:
[
  {{"action": "turn_on_tap", "duration": 3}},
  {{"action": "wet_hands", "duration": 3}},
  {{"action": "rub_hands", "duration": 4}}
]
"""

class TemporalScheduler:
    """
    Assigns start and end timestamps to a sequence of actions.
    Uses In-Context Learning by feeding dataset averages into the Groq LLM.
    """
    def __init__(self, groq_api_key: str, dataset_path: str = "handwash_dataset.csv"):
        print("[Scheduler] Initializing...")
        self.llm = GroqLLM(api_key=groq_api_key)
        self.dataset_path = dataset_path
        self.averages = self._compute_averages()
        print("[Scheduler] Ready.\n")
        
    def _compute_averages(self) -> Dict[str, float]:
        """Reads the CSV and calculates the mean duration for each action."""
        print(f"[Scheduler] Analyzing dataset: {self.dataset_path}")
        durations = defaultdict(list)
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        start = float(row["start_time"])
                        end = float(row["end_time"])
                        action = row["action"].strip().lower()
                        durations[action].append(end - start)
                    except (ValueError, KeyError):
                        continue
        except FileNotFoundError:
            print(f"[Scheduler] Warning: Dataset {self.dataset_path} not found. Proceeding without reference averages.")
            return {}
            
        averages = {action: round(sum(times) / len(times), 2) for action, times in durations.items()}
        print(f"[Scheduler] Found {len(averages)} reference actions in dataset.")
        return averages

    def _format_averages(self) -> str:
        """Formats the averages into a string for the LLM prompt."""
        if not self.averages:
            return "No dataset averages available."
        lines = [f"- {action}: {avg}s" for action, avg in self.averages.items()]
        return "\n".join(lines)

    def schedule(self, actions: List[str]) -> List[Dict[str, Any]]:
        """
        Takes a list of action strings and returns a scheduled timeline
        with start_time, end_time, and duration for each action.
        """
        if not actions:
            return []
            
        system_prompt = SCHEDULER_SYSTEM.format(dataset_averages=self._format_averages())
        user_prompt = json.dumps(actions)
        
        # Call LLM
        raw_response = self.llm.call(system_prompt, user_prompt)
        
        # Parse JSON
        raw_response = re.sub(r"```(?:json)?", "", raw_response).strip()
        try:
            durations_list = json.loads(raw_response)
        except json.JSONDecodeError:
            print("[Scheduler] Error: LLM returned invalid JSON. Falling back to default 3s durations.")
            durations_list = [{"action": action, "duration": 3.0} for action in actions]
            
        # Build timeline
        timeline = []
        current_time = 0.0
        
        # Create a dict for easy lookup to map LLM response back to exact input actions
        duration_map = {item.get("action"): item.get("duration", 3) for item in durations_list if isinstance(item, dict)}
        
        for action in actions:
            dur = duration_map.get(action, 3)
            try:
                dur = int(round(float(dur)))
            except (ValueError, TypeError):
                dur = 3
                
            # Cap duration between 1 and 8 seconds for video generation models
            dur = max(1, min(dur, 8))
                
            timeline.append({
                "step": action,
                "start": int(round(current_time)),
                "end": int(round(current_time + dur)),
                "duration": dur
            })
            current_time += dur
            
        return timeline
