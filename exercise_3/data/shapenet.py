from pathlib import Path
import json

import numpy as np
import torch


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 8
    dataset_sdf_path = Path("exercise_3/data/shapenet_dim32_sdf")  # path to voxel data
    dataset_df_path = Path("exercise_3/data/shapenet_dim32_df")  # path to voxel data
    class_name_mapping = json.loads(Path("exercise_3/data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split):
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.truncation_distance = 3

        self.items = Path(f"exercise_3/data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        sdf_id, df_id = self.items[index].split(' ')

        input_sdf = ShapeNet.get_shape_sdf(sdf_id)
        target_df = ShapeNet.get_shape_df(df_id)

        # TODO Apply truncation to sdf and df
        input_sdf = input_sdf.clip(-self.truncation_distance, self.truncation_distance)
        target_df = target_df.clip(-self.truncation_distance, self.truncation_distance)
        
        # TODO Stack (distances, sdf sign) for the input sdf
        input_sdf = np.stack((input_sdf, np.sign(input_sdf)), axis=0)

        # TODO Log-scale target df
        target_df = np.log(target_df + 1)

        return {
            'name': f'{sdf_id}-{df_id}',
            'input_sdf': input_sdf,
            'target_df': target_df
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        # TODO Move batch to device
        batch['input_sdf'] = batch['input_sdf'].to(device)
        batch['target_df'] = batch['target_df'].to(device)
        
    
    @staticmethod
    def get_shape_sdf(shapenet_id):
        #**Hint**: An easy way to load the data from `.sdf` and `.df` files is to use `np.fromfile`. First, load the dimensions, then the data, 
        #then reshape everything into the shape you loaded in the beginning. Make sure you get the datatypes and byte offsets right! 
        #If you are using the zip version of the dataset as explained above, you should use `np.frombuffer` instead of `np.fromfile` to load from the `data`-buffer. The syntax is identical.

        # Load the dimensions
        sdf_dims = np.fromfile(ShapeNet.dataset_sdf_path / f"{shapenet_id}.sdf", dtype=np.uint64, count=3)
        sdf_dims = sdf_dims.astype(np.uint64)
        sdf_dims = sdf_dims.reshape(3)

        # Load the data
        sdf_data = np.fromfile(ShapeNet.dataset_sdf_path /  f"{shapenet_id}.sdf", dtype=np.float32, offset=24) # size(uint64) = 8 bytes
        sdf_data = sdf_data.astype(np.float32)
    
        # Reshape the data
        sdf = sdf_data.reshape(sdf_dims[0], sdf_dims[1], sdf_dims[2])
        return sdf

    @staticmethod
    def get_shape_df(shapenet_id):
        # TODO implement df data loading
        # Load the dimensions
        df_dims = np.fromfile(ShapeNet.dataset_df_path / f"{shapenet_id}.df", dtype=np.uint64, count=3)
        df_dims = df_dims.astype(np.uint64)
        df_dims = df_dims.reshape(3)

        # Load the data
        df_data = np.fromfile(ShapeNet.dataset_df_path /  f"{shapenet_id}.df", dtype=np.float32, offset=24)
        df_data = df_data.astype(np.float32)

        # Reshape the data
        df = df_data.reshape(df_dims[0], df_dims[1], df_dims[2])
        return df
