from exercise_3.inference.infer_deepsdf import InferenceHandlerDeepSDF
import argparse
import torch
import os 
from exercise_3.data.shape_implicit import ShapeImplicit
from exercise_3.util.visualization import visualize_and_save_mesh, visualize_and_save_pointcloud


def main(): 

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--category", help="Specify a category from ['chair', 'picture_frame_painting', 'sofa', 'rug']", required=True)
    parser.add_argument("--experiment_name", help="Specify a category from ['chair', 'picture_frame_painting', 'sofa', 'rug']", required=True)
    parser.add_argument("--experiment_type", help="Specify a category from ['vanilla', 'pe', 'vn']", required=True)
    parser.add_argument("--is_overfit", help="Specify the extent of the experiment ", required=False, default=False)
    parser.add_argument("--shape_id", help="Specify a category from ['vanilla', 'pe', 'vn']", required=True)
    parser.add_argument("--vis_pointcloud", help="Specify a category from ['vanilla', 'pe', 'vn']", default=False, required=False)

                        
    args = parser.parse_args()
    shape_class = args.category
    experiment_name = args.experiment_name
    experiment_type = args.experiment_type
    shape_id = args.shape_id
    vis_pointcloud = args.vis_pointcloud

    save_path = f'/home/atasoy/project/dsdf/exercise_3/experiment_visuals/{experiment_name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device('cuda:0')  # change this to cpu if you're not using a gpu
    inference_handler = InferenceHandlerDeepSDF(256, f'/home/atasoy/project/dsdf/exercise_3/runs/{experiment_name}', device)
    points, sdf = ShapeImplicit.get_all_sdf_samples(shape_id, shape_class)

    if vis_pointcloud:
        inside_points = points[sdf[:, 0] < 0, :].numpy()
        outside_points = points[sdf[:, 0] > 0, :].numpy()
        # visualize observed points; you'll observe that the observations are very complete
        print('Observations with negative SDF (inside)')
        visualize_and_save_pointcloud(inside_points, 0.025, flip_axes=True)
        print('Observations with positive SDF (outside)')
        visualize_and_save_pointcloud(outside_points, 0.025, flip_axes=True)

    # Reconstruct as a mesh
    vertices, faces = inference_handler.reconstruct(points, sdf, 800)
    # visualize
    visualize_and_save_mesh(f'{save_path}/{shape_id}',vertices, faces, flip_axes=False)


if __name__ == "__main__": 

    main()