# Generate the train-overfit-val split for each category
# Take 1 shape for overfit, 0.9 for train, 0.1 for val
# Write them in overfit.txt train.txt val.txt
# Only write the shape name without extension
import random
import os
# Set the random seed
random.seed(42)

# Define the variable categories_with_more_than_100_shapes
# Iterate through each category
categories = ['sofa']
#categories = ['chair', 'rug', 'sofa', 'picture_frame_painting']

for category in categories:
    # Get the list of shapes for the current category
    shapes = []
    #Check if the path is a directory
    if os.path.isdir(f'./data/{category}'):
        shapes = os.listdir(f'./data/{category}')

    # Remove every file that is not a directory
    for shape in shapes:
        if not os.path.isdir(f'./data/{category}/{shape}'):
            os.remove(f'./data/{category}/{shape}')

    # Do not include anything that is not a directory
    shapes = [shape for shape in shapes if os.path.isdir(f'./data/{category}/{shape}')]

    # Shuffle the list of shapes
    random.shuffle(shapes)
    
    # Split the list of shapes into overfit, train, and val
    num_shapes = len(shapes)
    num_overfit = 1
    num_train = int(0.9 * (num_shapes - num_overfit))
    num_val = num_shapes - num_overfit - num_train

    print(f'Category: {category}')
    print(f'Number of shapes: {num_shapes}')
    print(f'Number of overfit: {num_overfit}')
    print(f'Number of train: {num_train}')
    print(f'Number of val: {num_val}')
    
    
    # Write the shape names to the respective files
    with open(f'/cluster/51/ataatasoy/project/dsdf/exercise_3/data/splits/{category}/overfit.txt', 'w') as f:
        f.write('\n'.join(shapes[:num_overfit]))
    
    with open(f'/cluster/51/ataatasoy/project/dsdf/exercise_3/data/splits/{category}/train.txt', 'w') as f:
        f.write('\n'.join(shapes[num_overfit:num_overfit+num_train]))
    
    with open(f'/cluster/51/ataatasoy/project/dsdf/exercise_3/data/splits/{category}/val.txt', 'w') as f:
        f.write('\n'.join(shapes[num_overfit+num_train:]))