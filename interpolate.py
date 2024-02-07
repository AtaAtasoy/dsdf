from exercise_3.data.shape_implicit import ShapeImplicit
from exercise_3.util.visualization import visualize_mesh
from exercise_3.inference.infer_deepsdf import InferenceHandlerDeepSDF
from exercise_3.util.mesh_collection_to_gif import  meshes_to_gif
from exercise_3.util.misc import show_gif
import torch
from pathlib import Path
import argparse
import os

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--category", help="Specify a category from ['chair', 'picture_frame_painting', 'sofa', 'rug']", required=True)
    parser.add_argument("--experiment_name", help="Specify a category from ['chair', 'picture_frame_painting', 'sofa', 'rug']", required=True)
    parser.add_argument("--experiment_type", help="Specify a category from ['vanilla', 'pe', 'vn']", required=True)
    parser.add_argument("--is_overfit", help="Specify the extent of the experiment ", required=False, default=False)
    parser.add_argument("--shape_id_1", help="Specify a category from ['vanilla', 'pe', 'vn']", required=True)
    parser.add_argument("--shape_id_2", help="Specify a category from ['vanilla', 'pe', 'vn']", required=True)
    parser.add_argument("--vis_pointcloud", help="Specify a category from ['vanilla', 'pe', 'vn']", default=False, required=False)

    args = parser.parse_args()
    shape_class = args.category
    experiment_name = args.experiment_name
    experiment_type = args.experiment_type
    shape_id_1 = args.shape_id_1
    shape_id_2 = args.shape_id_2
    vis_pointcloud = args.vis_pointcloud


    inference_handler = InferenceHandlerDeepSDF(256, f'/home/atasoy/project/dsdf/exercise_3/runs/{experiment_name}', torch.device('cuda:0'))
    # interpolate; also exports interpolated meshes to disk
    inference_handler.interpolate(shape_class, shape_id_1, shape_id_2, 60)

    os.environ["PYOPENGL_PLATFORM"] = "egl"

    # create list of meshes (just exported) to be visualized
    mesh_paths = sorted([x for x in Path(f"dsdf/exercise_3/runs/{experiment_name}/interpolation").iterdir() if int(x.name.split('.')[0].split("_")[1]) == 0], key=lambda x: int(x.name.split('.')[0].split("_")[0]))
    mesh_paths = mesh_paths + mesh_paths[::-1]

    # create a visualization of the interpolation process
    save_path = f'dsdf/exercise_3/experiment_visuals/{experiment_name}'
    meshes_to_gif(mesh_paths, f"{save_path}/{shape_id_1}_{shape_id_2}_latent_interp.gif", 20)


if __name__ == "__main__": 

    main()