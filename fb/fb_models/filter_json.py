import json
from collections import OrderedDict
import random

# Load your input JSON file
input_path = "PATH TO YOUR FULL DATASET" # Make sure to use the filtered file when you've already run this before.

with open(input_path, "r") as f:
    data = json.load(f)

# Collect example(s) per unique error_tag
unique_by_tag = OrderedDict()
used_ids = set()
# tag_counts = {} # <= For multi-shot

for item in data:
    tag = item.get("error_tag")

    ### one instance per error tag (one-shot)
    if tag and tag not in unique_by_tag:
        unique_by_tag[tag] = item
        used_ids.add(id(item))  # use Python's object ID to identify selected entries

    ### multiple instances per error tag (multi-shot)
    # if tag:
    #     tag_counts[tag] = tag_counts.get(tag, 0) + 1
    #     if tag_counts[tag] <= 2: ## change the value to align with n-shot
    #         unique_by_tag[tag + f"_{tag_counts[tag]}"] = item
    #         used_ids.add(id(item))  # use Python's object ID to identify selected entries

# Create filtered list for n-shot data
nshot_data = list(unique_by_tag.values())



# Count how many examples per error_tag in n-shot_data
from collections import defaultdict

tag_freq = defaultdict(int)
for item in nshot_data:
    tag = item["error_tag"]
    base_tag = tag.split("_")[0] if "_" in tag else tag  # remove suffix like "_1", "_2"
    tag_freq[base_tag] += 1

print("\nBreakdown of examples per error tag in oneshot_data:")
for tag, count in sorted(tag_freq.items()):
    print(f"{tag}: {count} examples")



# Create a list of remaining examples
rest_data = [item for item in data if id(item) not in used_ids]

# Save both JSON files
output_nshot = "OUTPUT (N-SHOT) PATH"
output_rest = "OUTPUT (REST) PATH"

with open(output_nshot, "w") as f:
    json.dump(nshot_data, f, indent=2, ensure_ascii=False)

with open(output_rest, "w") as f:
    json.dump(rest_data, f, indent=2, ensure_ascii=False)

print(f"Saved {len(nshot_data)} unique examples → {output_nshot}")
print(f"Saved {len(rest_data)} remaining examples → {output_rest}")