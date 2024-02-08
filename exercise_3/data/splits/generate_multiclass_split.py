import json
import random
from pathlib import Path
import shutil
import os

# Your existing code for creating dictionaries
sofa_items = os.listdir('/workspace/project/data/sofa')
bed_items = os.listdir('/workspace/project/data/bed')

# Get the minimum number of items in the classes
min_items = min(len(sofa_items), len(bed_items))

sofa_items = random.sample(sofa_items, min_items)
bed_items = random.sample(bed_items, min_items)

# Create a {item: class} dictionary for the items
sofa_dict = {}
for item in sofa_items:
    sofa_dict[item] = 0
    
bed_dict = {}
for item in bed_items:
    bed_dict[item] = 1
    
# Have a combined dictionary
combined_dict = {**sofa_dict, **bed_dict}

all_items = list(combined_dict.keys())

# Shuffle the items
random.shuffle(all_items)

# Split the items into train and val and overfit
num_shapes = len(all_items)
num_overfit = 1
num_train = int(0.9 * (num_shapes - num_overfit))
num_val = num_shapes - num_overfit - num_train

print(f'Number of shapes: {num_shapes}')
print(f'Number of overfit: {num_overfit}')
print(f'Number of train: {num_train}')
print(f'Number of val: {num_val}')

# Write the shape names to the respective files, create the text file if it does not exist. Also 
with open(f'/workspace/project/dsdf/exercise_3/data/splits/sofa_bed/overfit.txt', 'w') as f:
    for item in all_items[:num_overfit]:
        f.write(f'{item} {combined_dict[item]}\n')
    

with open(f'/workspace/project/dsdf/exercise_3/data/splits/sofa_bed/train.txt', 'w') as f:
   for item in all_items[num_overfit:num_overfit+num_train]:
        f.write(f'{item} {combined_dict[item]}\n')


with open(f'/workspace/project/dsdf/exercise_3/data/splits/sofa_bed/val.txt', 'w') as f:
       for item in all_items[num_overfit+num_train:]:
        f.write(f'{item} {combined_dict[item]}\n')

# Copy the files to the respective folders
for item in all_items:
    if item in sofa_items:
        shutil.copytree(f'/workspace/project/data/sofa/{item}', f'/workspace/project/data/sofa_bed/{item}')
    elif item in bed_items:
        shutil.copytree(f'/workspace/project/data/bed/{item}', f'/workspace/project/data/sofa_bed/{item}')