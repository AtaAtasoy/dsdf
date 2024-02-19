import os
from pathlib import Path

sofa_bed_train = Path('/home/atasoy/project/dsdf/exercise_3'+ '/data/splits/' + 'sofa_bed' + '/train.txt').read_text().splitlines()
sofa_bed_val = Path('/home/atasoy/project/dsdf/exercise_3'+ '/data/splits/' + 'sofa_bed' + '/val.txt').read_text().splitlines()
sofa_bed_overfit = Path('/home/atasoy/project/dsdf/exercise_3'+ '/data/splits/' + 'sofa_bed' + '/overfit.txt').read_text().splitlines()


train_set = []
val_set = []
overfit_set = []

for line in sofa_bed_train:
    item = line.split(' ')[0]
    category = line.split(' ')[1]
    if category == "1":
        train_set.append(item)
        
for line in sofa_bed_val:
    item = line.split(' ')[0]
    category = line.split(' ')[1]
    if category == "1":
        val_set.append(item)
        
for line in sofa_bed_overfit:
    item = line.split(' ')[0]
    category = line.split(' ')[1]
    if category == "1":
        overfit_set.append(item)

# write the train, val and overfit sets to a file
with open('train_set.txt', 'w') as f:
    for item in train_set:
        f.write("%s\n" % item)
        
with open('val_set.txt', 'w') as f:
    for item in val_set:
        f.write("%s\n" % item)
        
with open('overfit_set.txt', 'w') as f:
    for item in overfit_set:
        f.write("%s\n" % item)