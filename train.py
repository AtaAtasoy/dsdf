from exercise_3.training import train_deepsdf
from exercise_3.util.model import summarize_model
from exercise_3.util.visualization import visualize_mesh, visualize_pointcloud
import argparse

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--category", help="Specify a category from ['chair', 'sofa', 'bed', 'multiclass']", required=True)
    parser.add_argument("--experiment_name", help="Give a name for your experiment i.e bed_generalization_vanilla_1", required=True)
    parser.add_argument("--experiment_type", help="Specify a category from ['vanilla', 'pe', 'vn', 'multiclass']", required=True)
    parser.add_argument("--is_overfit", help="Specify the extent of the experiment ", required=False, default=False, type=bool)
    parser.add_argument("--num_points", help="Specify the number of points to sample for sdf values per shape", required=False, default=4096, type=int)
                        
    args = parser.parse_args()
    shape_class = args.category
    experiment_name = args.experiment_name
    experiment_type = args.experiment_type
    is_overfit = args.is_overfit


    generalization_config = {
        'experiment_name': experiment_name,
        'experiment_type': experiment_type,
        'shape_class': shape_class,
        'number_of_classes': 2,
        'device': 'cuda:0',  # run this on a gpu for a reasonable training time
        'is_overfit': is_overfit,
        'num_sample_points': 4096, # you can adjust this such that the model fits on your gpu
        'latent_code_length': 256,
        'class_embedding_length': 128,
        'batch_size': 1,
        'resume_ckpt': None,
        'learning_rate_model': 0.0005,
        'learning_rate_code': 0.001,
        'learning_rate_class_code': 0.005,
        'lambda_code_regularization': 0.0001,
        'max_epochs': 500,  # not necessary to run for 2000 epochs if you're short on time, at 500 epochs you should start to see reasonable results
        'print_every_n': 50,
        'visualize_every_n': 5000,
        'num_encoding_functions': 4,
    }

    train_deepsdf.main(generalization_config)


if __name__ == "__main__": 

    main()
    