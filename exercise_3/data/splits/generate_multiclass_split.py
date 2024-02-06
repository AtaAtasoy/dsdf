import json
import random
from pathlib import Path
import shutil
import os

# Your existing code for creating dictionaries
sofa_items = Path('/workspace/project/dsdf/exercise_3'+ '/data/splits/' + 'sofa' + '/train.txt').read_text().splitlines()[:300]
bed_items = Path('/workspace/project/dsdf/exercise_3'+ '/data/splits/' + 'bed' + '/train.txt').read_text().splitlines()[:300]

# Create a {item: class} dictionary for the items
sofa_dict = {}
for item in sofa_items:
    sofa_dict[item] = 0
    
bed_dict = {}
for item in bed_items:
    bed_dict[item] = 1
    
# Have a combined dictionary
combined_dict = {**sofa_dict, **bed_dict}

# Dump this dictionary to a JSON file
output_file_path = '/workspace/project/dsdf/exercise_3/data/splits/multiclass/shuffled_items.json'
with open(output_file_path, 'w') as output_file:
    json.dump(combined_dict, output_file, indent=2)

all_items = list(combined_dict.keys())

# Shuffle the items
random.shuffle(all_items)

# Split the items into train and val and overfit
num_shapes = len(all_items)
num_overfit = 2
num_train = int(0.9 * (num_shapes - num_overfit))
num_val = num_shapes - num_overfit - num_train

print(f'Number of shapes: {num_shapes}')
print(f'Number of overfit: {num_overfit}')
print(f'Number of train: {num_train}')
print(f'Number of val: {num_val}')


# Write the shape names to the respective files, create the text file if it does not exist. Also 
with open(f'/workspace/project/dsdf/exercise_3/data/splits/multiclass/overfit.txt', 'w') as f:
    for item in all_items[:num_overfit]:
        f.write(f'{item} {combined_dict[item]}\n')
    

with open(f'/workspace/project/dsdf/exercise_3/data/splits/multiclass/train.txt', 'w') as f:
   for item in all_items[num_overfit:num_overfit+num_train]:
        f.write(f'{item} {combined_dict[item]}\n')


with open(f'/workspace/project/dsdf/exercise_3/data/splits/multiclass/val.txt', 'w') as f:
       for item in all_items[num_overfit+num_train:]:
        f.write(f'{item} {combined_dict[item]}\n')

