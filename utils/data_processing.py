import json
from datetime import datetime

def save_result_to_json(results):
    # Saves results as a JSON file with a timestamp.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"data/output/output_{timestamp}.json"
    with open(filename, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Results saved to {filename}")
