from exercise_3.training import train_deepsdf
from exercise_3.util.model import summarize_model
from exercise_3.util.visualization import visualize_mesh, visualize_pointcloud
import argparse

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--category", help="Specify a category from [, 'sofa', 'bed', 'sofa_bed, sofa_bed_disjoint']", required=True)
    parser.add_argument("--experiment_type", help="Specify a category from ['single_class', 'multi_class']", required=True)
    parser.add_argument("--is_overfit", help="Specify the extent of the experiment ", required=False, default=False, type=bool)
    parser.add_argument("--num_points", help="Specify the number of points to sample for sdf values per shape", required=False, default=4096, type=int)
    parser.add_argument("--epochs", help="Specify the number of epochs to train the model", required=False, default=500, type=int)
    parser.add_argument("--class_embedding_length", help="Specify the length of the class embedding", required=False, default=128, type=int)
    parser.add_argument("--latent_code_length", help="Specify the length of the latent code", required=False, default=256, type=int)
    parser.add_argument("--num_encoding_functions", help="Specify the number of encoding functions. Default 2", required=False, default=2, type=int)
    parser.add_argument("--number_of_classes", help="Specify the number of classes to train the model on", required=False, default=1, type=int)
    parser.add_argument("--resume", help="Resume an interrupted experiment", required=False, default=False, type=bool)
    
    args = parser.parse_args()
    shape_class = args.category
    experiment_type = args.experiment_type
    is_overfit = args.is_overfit
    num_points = args.num_points
    epochs = args.epochs
    latent_code_length = args.latent_code_length
    class_embedding_length = args.class_embedding_length
    num_encoding_functions = args.num_encoding_functions
    number_of_classes = args.number_of_classes
    resume = args.resume
    
    experiment_name = f'{shape_class}_{experiment_type}_{num_points}points__{epochs}epochs__{latent_code_length}latent_code__{class_embedding_length}class_embedding_length__{num_encoding_functions}enc'
    #experiment_name = 'bed_single_class_4096points__1000epochs__256latent_code__128class_embedding_length__2enc'
    
    generalization_config = {
        'experiment_name': experiment_name,
        'experiment_type': experiment_type,
        'shape_class': shape_class,
        'device': 'cuda:0',  # run this on a gpu for a reasonable training time
        'number_of_classes': number_of_classes,
        'is_overfit': is_overfit,
        'num_sample_points': num_points, # you can adjust this such that the model fits on your gpu
        'latent_code_length': latent_code_length,
        'class_embedding_length': class_embedding_length,
        'batch_size': 1,
        'resume_ckpt': resume,
        'learning_rate_model': 0.0005,
        'learning_rate_code': 0.001,
        'learning_rate_class_code': 0.001,
        'lambda_code_regularization': 0.0001,
        'max_epochs': epochs,  # not necessary to run for 2000 epochs if you're short on time, at 500 epochs you should start to see reasonable results
        'print_every_n': 50,
        'visualize_every_n': 5000,
        'num_encoding_functions': num_encoding_functions,
    }
        
    train_deepsdf.main(generalization_config)

if __name__ == "__main__": 
    main()
    
