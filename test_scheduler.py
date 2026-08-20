import os
from scheduler import TemporalScheduler

API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError("Set GROQ_API_KEY before running the scheduler tests.")

def test_scheduler():
    scheduler = TemporalScheduler(groq_api_key=API_KEY)
    
    # Let's test with the output from a parser run
    # 1. Standard Handwash (should closely match dataset averages)
    handwash_actions = [
        "turn_on_tap",
        "wet_your_hands_with_water",
        "apply_soap",
        "rub_your_hands",
        "rinse_your_hands_with_water",
        "turn_off_tap",
        "dry_your_hands_with_towel"
    ]
    
    # 2. General Cooking Task (should use general knowledge)
    cooking_actions = [
        "chop_onions",
        "fry_onions_in_oil",
        "add_salt_and_pepper_to_onions",
        "pour_water_into_pot",
        "boil_onions_for_minutes"
    ]
    
    # 3. Tea Making Task
    tea_actions = [
        "boil_water",
        "place_tea_bag_in_cup",
        "pour_hot_water_into_cup",
        "wait_a_few_minutes",
        "remove_tea_bag_from_cup",
        "add_sugar_to_tea"
    ]

    tests = {
        "Standard Handwashing (Dataset aligned)": handwash_actions,
        "Cooking Task (General knowledge)": cooking_actions,
        "Making Tea (General knowledge)": tea_actions
    }
    
    for test_name, actions in tests.items():
        print(f"\\n{'='*60}")
        print(f" TEST: {test_name}")
        print(f"{'='*60}")
        
        timeline = scheduler.schedule(actions)
        
        # Pretty print the timeline
        print(f"{'Start':>5}s -> {'End':>5}s  ({'Dur':>4}s)  |  Action")
        print("-" * 60)
        for item in timeline:
            print(f"{item['start']:5d}s -> {item['end']:5d}s  ({item['duration']:4d}s)  |  {item['step']}")

if __name__ == "__main__":
    test_scheduler()
