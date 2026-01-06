import json
import random

# CONFIG
INPUT_FILE = "datasets/tailwind_augmented.json" # Use your augmented (diverse) file
OUTPUT_FILE = "datasets/tailwind_small.json"
SAMPLES_PER_CLASS = 50  # 40 * 12 classes = 480 total samples (Very Small!)

def create_subset():
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group by label to ensure balance
    grouped = {}
    for item in data:
        label = item["label"]
        if label not in grouped: grouped[label] = []
        grouped[label].append(item)

    # Select random subset
    subset = []
    print(f"\nSelecting {SAMPLES_PER_CLASS} samples per class...")
    
    for label, items in grouped.items():
        if len(items) < SAMPLES_PER_CLASS:
            print(f"Warning: {label} only has {len(items)} items. Taking all.")
            subset.extend(items)
        else:
            subset.extend(random.sample(items, SAMPLES_PER_CLASS))

    # Shuffle final list
    random.shuffle(subset)

    print(f"Writing {len(subset)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    create_subset()