import os
import json
from env_loader import load_dotenv_file

load_dotenv_file()
API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError("Set GROQ_API_KEY before running the pipeline.")

from parser import InstructionParser
from scheduler import TemporalScheduler
from test_parser import tests

def run_pipeline():
    print("Initializing Full Pipeline (Parser + Scheduler)...")
    parser = InstructionParser(groq_api_key=API_KEY)
    scheduler = TemporalScheduler(groq_api_key=API_KEY)
    
    results = []
    
    print("\n" + "="*65)
    print(" RUNNING END-TO-END PIPELINE")
    print("="*65)
    
    for label, prompt in tests:
        print(f"\nProcessing: {label}")
        print(f"Input: {prompt!r}")
        
        try:
            # Stage 1 & 2: Parse
            parse_result = parser.parse(prompt, verbose=False)
            steps = [a.step for a in parse_result.actions]
            
            # Stage 3: Schedule
            timeline = scheduler.schedule(steps)
            
            # Store
            results.append({
                "test": label,
                "input": prompt,
                "parsed_steps": steps,
                "timeline": timeline,
                "warnings": parse_result.warnings,
                "status": "PASS"
            })
            print(f" -> Scheduled {len(timeline)} clips. Total duration: {timeline[-1]['end'] if timeline else 0}s")
            
        except Exception as e:
            print(f" -> ERROR: {e}")
            results.append({
                "test": label,
                "input": prompt,
                "parsed_steps": [],
                "timeline": [],
                "warnings": [],
                "status": f"ERROR: {e}"
            })

    # Save to JSON
    output_file = "pipeline_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✅ Pipeline complete! Results saved to {output_file}")

if __name__ == "__main__":
    run_pipeline()
