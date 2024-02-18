import os 

single_class_sofa_validation_path = os.path.join('sofa', 'val.txt')
multi_class_validation_path = os.path.join('multiclass', 'val.txt')

single_class_sofa_ids = []
multi_class_ids = []

with open(single_class_sofa_validation_path, 'r') as f:
    single_class_sofa_ids = f.read().splitlines()
    
with open(multi_class_validation_path, 'r') as f:
    line = f.read().splitlines()
    multi_class_ids = [l.split(' ')[0] for l in line if l.split(' ')[1] == '0']
    
# find intersections
intersection = list(set(single_class_sofa_ids).intersection(multi_class_ids))
print(single_class_sofa_ids)
print(multi_class_ids)

print(intersection)
    