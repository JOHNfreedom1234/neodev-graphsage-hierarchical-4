import json
from collections import Counter

# Load your file
with open("datasets/tailwind_augmented.json", "r") as f:
    data = json.load(f)

print(f"Total Samples: {len(data)}")

seen_labels = set()
duplicates_found = False

print("\n--- UNIQUENESS REPORT ---")

for target_label in sorted(list(set(d["label"] for d in data))):
    # Collect all items for this specific label
    items_of_label = [d for d in data if d["label"] == target_label]
    
    # Store unique structures
    unique_structures = set()
    
    for item in items_of_label:
        # Flexible key access (The Fix)
        root = item.get("dom") or item.get("content") or item.get("contents")
        
        # Create a string "fingerprint" of the structure to compare them
        # We sort keys to ensure {"a":1, "b":2} is same as {"b":2, "a":1}
        fingerprint = json.dumps(root, sort_keys=True)
        unique_structures.add(fingerprint)
    
    total = len(items_of_label)
    unique = len(unique_structures)
    
    print(f"Label: {target_label:<15} | Total: {total:<4} | Unique Variations: {unique}")
    
    if unique == 1 and total > 1:
        duplicates_found = True

print("-" * 60)
if duplicates_found:
    print("🚨 CONCLUSION: The dataset contains massive duplication.")
    print("   This explains why the CNN got 100% accuracy.")
    print("   It simply memorized the 1 unique pattern for each class.")
else:
    print("✅ The dataset appears diverse.")