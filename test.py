from pathlib import Path
import numpy as np
from exercise_3.inference.infer_deepsdf import InferenceHandlerDeepSDF
import trimesh
import torch
import argparse

from exercise_3.data.shape_implicit import ShapeImplicit
from exercise_3.util.visualization import visualize_mesh, visualize_pointcloud
from exercise_3.model.deepsdf import DeepSDFDecoder
from exercise_3.util.model import summarize_model
from exercise_3.training import train_deepsdf
from exercise_3.tests.chamferdist import get_chamfer_distance, get_sinkhorn_distance
import warnings
warnings.filterwarnings("ignore")


def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--category", help="Specify a category from ['chair', 'picture_frame_painting', 'sofa', 'rug']", required=True)
    parser.add_argument("--experiment_name", help="Specify a category from ['chair', 'picture_frame_painting', 'sofa', 'rug']", required=True)
    parser.add_argument("--experiment_type", help="Specify a category from ['vanilla', 'pe', 'vn']", required=True)

    args = parser.parse_args()

    experiment_type = args.experiment_type
    experiment_name = args.experiment_name
    shape_class = args.category

    shape_id_to_class_name = {0: 'sofa', 1: 'bed'}

    device = torch.device('cuda:0')  # change this to cpu if you're not using a gpu

    #inference_handler = InferenceHandlerDeepSDF(256, f"/home/atasoy/project/dsdf/exercise_3/runs/{experiment_name}", device, num_encoding_functions=2, experiment_type=experiment_type)


    #Get the val.txt file for the shape class
    val_file = f'/home/atasoy/project/dsdf/exercise_3/data/splits/sofa_bed/val.txt'
    shapes = Path(val_file).read_text().splitlines()

    # Split the lines in shape_ids to get the shape_id and the class_id
    shape_ids = [shape_ids.split(' ')[0] for shape_ids in shapes]
    class_ids = [shape_ids.split(' ')[1] for shape_ids in shapes]
    num_val_shapes = len(shape_ids)
    chamfer_dist = 0
    #sinkhorn_dist = 0

    for shape_id, class_id in zip(shape_ids, class_ids):
            class_id = int(class_id)
            inference_handler = InferenceHandlerDeepSDF(256, f'/home/atasoy/project/dsdf/exercise_3/runs/{experiment_name}', device, class_idx=class_id, experiment_type=experiment_type, experiment_class= 'sofa_bed')
        #try:
            print(f"Processing shape {shape_id}")
            mesh_path = f'/mnt/hdd/atasoy_dataset/ml3d/sofa_bed/{shape_id}/mesh.obj'
            mesh = trimesh.load(mesh_path)
            # Calculate the centroid of the mesh
            centroid = mesh.centroid
            # Translate the mesh so that the centroid is at the origin
            mesh = mesh.apply_translation(-centroid)

            # Step 2: Scale the mesh to fit within a unit sphere
            # Calculate the furthest distance from the origin to any vertex
            max_distance = np.max(np.linalg.norm(mesh.vertices, axis=1))
            # Calculate the scaling factor (the mesh will be scaled so that the furthest point is 1 unit away)
            scaling_factor = 1.0 / max_distance
            # Scale the mesh
            mesh = mesh.apply_scale(scaling_factor)

            file_path_orig = '/home/atasoy/project/dsdf/try_mesh_orig.obj'
            with open(file_path_orig, 'w') as file:
                mesh.export(file_obj=file, file_type='obj')

            
            points, sdf = ShapeImplicit.get_all_sdf_samples(shape_id, 'sofa_bed')
            vertices, faces = inference_handler.reconstruct(points, sdf, 800)
            ref = trimesh.Trimesh(vertices=vertices, faces=faces)
            # Calculate the centroid of the mesh
            centroid = ref.centroid
            # Translate the mesh so that the centroid is at the origin
            ref = ref.apply_translation(-centroid)

            # Step 2: Scale the mesh to fit within a unit sphere
            # Calculate the furthest distance from the origin to any vertex
            
            max_distance = np.max(np.linalg.norm(ref.vertices, axis=1))
            
            # Calculate the scaling factor (the mesh will be scaled so that the furthest point is 1 unit away)
            scaling_factor = 1.0 / max_distance
            # Scale the mesh
            ref = ref.apply_scale(scaling_factor)
            file_path = '/home/atasoy/project/dsdf/try_mesh.obj'
            with open(file_path, 'w') as file:
                ref.export(file_obj=file, file_type='obj')
            chamfer_dist += get_chamfer_distance(file_path_orig, file_path)
            print(f"Chamfer distance for shape {shape_id}, {chamfer_dist}")
            #sinkhorn_dist += get_sinkhorn_distance(mesh_path, file_path)
            #print(f"Chamfer distance for shape {shape_id}, {chamfer_dist}")
        #except ValueError:  #raised if `y` is empty.
        #        pass

    print(f"Mean Chamfer Distance: {chamfer_dist/num_val_shapes}, for experiment {experiment_name}")
    #print(f"Mean Sinkhorn Distance: {sinkhorn_dist/num_val_shapes}, for experiment {experiment_name}")



if __name__ == "__main__":
    main()