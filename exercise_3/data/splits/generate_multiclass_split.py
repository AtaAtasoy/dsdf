import json
import random
from pathlib import Path

# Your existing code for creating dictionaries
sofa_items = Path('/workspace/project/dsdf/exercise_3'+ '/data/splits/' + 'sofa' + '/train.txt').read_text().splitlines()[:300]
bed_items = Path('/workspace/project/dsdf/exercise_3'+ '/data/splits/' + 'bed' + '/train.txt').read_text().splitlines()[:300]
chair_items = Path('/workspace/project/dsdf/exercise_3'+ '/data/splits/' + 'chair' + '/train.txt').read_text().splitlines()[:300]

sofa_dict = {}
for item in sofa_items:
    sofa_dict[item] = 'sofa'

bed_dict = {}
for item in bed_items:
    bed_dict[item] = 'bed'

chair_dict = {}
for item in chair_items:
    chair_dict[item] = 'chair'

# Combine all dictionaries into a list
all_dicts = [sofa_dict, bed_dict, chair_dict]

# Shuffle the items within each dictionary
for d in all_dicts:
    items_list = list(d.items())
    random.shuffle(items_list)
    d.clear()
    d.update(items_list)

# Combine all items into a single list
# Insert as a dictionary with the item and its class
all_items = [{"item": item, "class": label} for d in all_dicts for item, label in d.items()]

# Shuffle the combined list
random.shuffle(all_items)

# Save the shuffled items as a JSON file
output_file_path = '/workspace/project/dsdf/exercise_3/data/splits/multiclass/shuffled_items.json'  # Specify the desired output file path
with open(output_file_path, 'w') as output_file:
    json.dump(all_items, output_file, indent=2)
