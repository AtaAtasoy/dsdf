from exercise_3.training import train_deepsdf

generalization_config = {
    'experiment_name': 'VNN_1000epoch_vanilla_bed',
    'device': 'cuda:0',  # run this on a gpu for a reasonable training time
    'is_overfit': False,
    'num_sample_points': 4096, # you can adjust this such that the model fits on your gpu
    'latent_code_length': 256,
    'batch_size': 1,
    'resume_ckpt': None,
    'learning_rate_model': 0.0005,
    'learning_rate_code': 0.001,
    'lambda_code_regularization': 0.0001,
    'max_epochs': 1000,  
    'print_every_n': 50,
    'visualize_every_n': 5000,
    'experiment_type': 'vanilla',
    'rotate_augment': False,
    'shape_class': 'bed',
    'num_encoding_functions': 2,
}

train_deepsdf.main(generalization_config)