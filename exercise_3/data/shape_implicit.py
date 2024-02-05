from pathlib import Path
import numpy as np
import torch
import trimesh
import os

from exercise_3.util.misc import remove_nans
from exercise_3.util.data_augmentations import apply_random_rotation

from .positional_encoding import positional_encoding

class ShapeImplicit(torch.utils.data.Dataset):
    """
    Dataset for loading deep sdf training samples
    """
    
    dataset_path = '/cluster/51/ataatasoy/project/data/'

    def __init__(self, shape_class, num_sample_points, split, num_encoding_functions=6, rotate_augment=False):
        """
        :param num_sample_points: number of points to sample for sdf values per shape
        :param split: one of 'train', 'val' or 'overfit' - for training, validation or overfitting split
        """
        super().__init__()
        assert split in ['train', 'val', 'overfit']

        self.num_sample_points = num_sample_points
        self.dataset_path = Path(f'{ShapeImplicit.dataset_path}/{shape_class}') # path to the sdf data for ShapeNetSem
        self.items = Path(f'/cluster/51/ataatasoy/project/dsdf/exercise_3/data/splits/{shape_class}/{split}.txt').read_text().splitlines()  # keep track of shape identifiers based on split
        self.pe_encoder = lambda x: positional_encoding(x, num_encoding_functions=num_encoding_functions)
        self.rotate_augment = rotate_augment

    def __getitem__(self, index):
        """
        PyTorch requires you to provide a getitem implementation for your dataset.
        :param index: index of the dataset sample that will be returned
        :return: a dictionary of sdf data corresponding to the shape. In particular, this dictionary has keys
             "name", shape_identifier of the shape
             "indices": index parameter
             "points": a num_sample_points x 3  pytorch float32 tensor containing sampled point coordinates
             "sdf", a num_sample_points x 1 pytorch float32 tensor containing sdf values for the sampled points
        """

        # get shape_id at index
        item = self.items[index]

        # get path to sdf data
        sdf_samples_path = f'{self.dataset_path}/{item}/sdf.npz'

        # read points and their sdf values from disk
        # TODO: Implement the method get_sdf_samples
        sdf_samples = self.get_sdf_samples(sdf_samples_path)

        points = sdf_samples[:, :3]
        if self.rotate_augment:
            points = apply_random_rotation(points=points)
        encoded_points = self.pe_encoder(points)
        sdf = sdf_samples[:, 3:]

        # truncate sdf values
        sdf_clamped = torch.clamp(sdf, -0.1, 0.1)

        return {
            "name": item,       # identifier of the shape
            "indices": index,   # index parameter
            "points": encoded_points,   # points (pos + PE Encoding), a tensor with shape num_sample_points x (3 + 3 * 2 * num_encoding_functions)
            "sdf": sdf_clamped  # sdf values, a tensor with shape num_sample_points x 1
        }

    def __len__(self):
        """
        :return: length of the dataset
        """
        # TODO: Implement
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch['points'] = batch['points'].to(device)
        batch['sdf'] = batch['sdf'].to(device)
        batch['indices'] = batch['indices'].to(device)

    def get_sdf_samples(self, path_to_sdf):
        """
        Utility method for reading an sdf file; the SDF file for a shape contains a number of points, along with their sdf values
        :param path_to_sdf: path to sdf file
        :return: a pytorch float32 torch tensor of shape (num_sample_points, 4) with each row being [x, y, z, sdf_value at xyz]
        """
        print(f'Loading {path_to_sdf}')
        npz = np.load(path_to_sdf)
        pos_tensor = remove_nans(torch.from_numpy(npz["pos"].astype(np.float32)))
        neg_tensor = remove_nans(torch.from_numpy(npz["neg"].astype(np.float32)))


        #print(f'Shape of pos_tensor: {pos_tensor.shape}')
        #print(f'Shape of neg_tensor: {neg_tensor.shape}')
        # TODO: Implement such that you return a pytorch float32 torch tensor of shape (self.num_sample_points, 4)
        # the returned tensor shoud have approximately self.num_sample_points/2 randomly selected samples from pos_tensor
        # and approximately self.num_sample_points/2 randomly selected samples from neg_tensor 

        pos_tensor = pos_tensor[torch.randperm(self.num_sample_points//2)]
        neg_tensor = neg_tensor[torch.randperm(self.num_sample_points//2)]
        #print(f'Sampled shape of pos_tensor: {pos_tensor.shape}')
        #print(f'Sampled shape of neg_tensor: {neg_tensor.shape}')
        
        samples = torch.cat([pos_tensor, neg_tensor])
        #print(f'Samples shape: {samples.shape}, {samples.dtype}')
        #print(f'Samples shape: {samples.shape}')
        # Hint: You can use torch.randperm to generate a random permutation of indices

        return samples

    @staticmethod
    def get_mesh(shape_id, shape_class):
        """
        Utility method for loading a mesh from disk given shape identifier
        :param shape_id: shape identifier for ShapeNet object
        :return: trimesh object representing the mesh
        """
        mesh_path = f'{ShapeImplicit.dataset_path}/{shape_class}/{shape_id}/mesh_simplified.obj'
        return trimesh.load(mesh_path, force='mesh')

    @staticmethod
    def get_all_sdf_samples(shape_id, shape_class):
        """
        Utility method for loading all points and their sdf values from disk
        :param shape_id: shape identifier for ShapeNet object
        :return: two torch float32 tensors, a Nx3 tensor containing point coordinates, and Nx1 tensor containing their sdf values
        """
        npz = np.load(ShapeImplicit.dataset_path / shape_class / shape_id / "sdf.npz")
        pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
        neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

        samples = torch.cat([pos_tensor, neg_tensor], 0)
        points = samples[:, :3]

        # trucate sdf values
        sdf = torch.clamp(samples[:, 3:], -0.1, 0.1)

        return points, sdf
