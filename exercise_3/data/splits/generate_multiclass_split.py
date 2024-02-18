import json
import random
from pathlib import Path
import shutil
import os

sofa_bed_train = Path('/home/atasoy/project/dsdf/exercise_3'+ '/data/splits/' + 'sofa_bed' + '/train.txt').read_text().splitlines()
sofa_bed_val = Path('/home/atasoy/project/dsdf/exercise_3'+ '/data/splits/' + 'sofa_bed' + '/val.txt').read_text().splitlines()
sofa_bed_overfit = Path('/home/atasoy/project/dsdf/exercise_3'+ '/data/splits/' + 'sofa_bed' + '/overfit.txt').read_text().splitlines()

dataset_src_path = Path('/mnt/hdd/atasoy_dataset/ml3d')
#dataset_src_path = Path('/home/atasoy/project/data')

# Create multiclass folder under dataset path
multiclass_path = dataset_src_path / 'sofa_bed'
multiclass_path.mkdir(exist_ok=True)

# Copy the sofas to the multiclass folder
for line in sofa_bed_train:
    item = line.split(' ')[0]
    category = line.split(' ')[1]
    if category == "0":
        shutil.copytree(dataset_src_path / 'sofa' / item, multiclass_path / item)
    elif category == "1":
        shutil.copytree(dataset_src_path / 'bed' / item, multiclass_path / item)
        
for line in sofa_bed_val:
    item = line.split(' ')[0]
    category = line.split(' ')[1]
    if category == "0":
        shutil.copytree(dataset_src_path / 'sofa' / item, multiclass_path / item)
    elif category == "1":
        shutil.copytree(dataset_src_path / 'bed' / item, multiclass_path / item)
        
for line in sofa_bed_overfit:
    item = line.split(' ')[0]
    category = line.split(' ')[1]
    if category == "0":
        shutil.copytree(dataset_src_path / 'sofa' / item, multiclass_path / item)
    elif category == "1":
        shutil.copytree(dataset_src_path / 'bed' / item, multiclass_path / item)
