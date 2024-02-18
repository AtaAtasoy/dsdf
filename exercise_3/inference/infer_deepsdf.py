import random
from pathlib import Path

import torch

from exercise_3.data.shape_implicit import ShapeImplicit
from exercise_3.model.deepsdf import DeepSDFDecoder
from exercise_3.util.misc import evaluate_model_on_grid

from exercise_3.data.positional_encoding import positional_encoding
from exercise_3.data.positional_encoding import positional_encoding

class InferenceHandlerDeepSDF:

    def __init__(self, latent_code_length, experiment, device, class_idx, num_encoding_functions=2, experiment_type='multi_class', class_embedding_length=128, experiment_class="multiclass"):
        """
        :param latent_code_length: latent code length for the trained DeepSDF model
        :param experiment: path to experiment folder for the trained model; should contain "model_best.ckpt" and "latent_best.ckpt"
        :param device: torch device where inference is run
        :param num_encoding_functions: number of encoding functions used in the model
        :param experiment_type: type of experiment, 'vanilla', 'pe' or 'multiclass'
        :param class_embedding_length: length of class embedding
        :param class_idx: index of the class for which inference is performed 
        :param experiment_class: class of the experiment
        """
        self.latent_code_length = latent_code_length
        self.experiment = Path(experiment)
        self.device = device
        self.truncation_distance = 0.01
        self.num_samples = 4096
        self.num_encoding_functions = num_encoding_functions
        self.pe_encoder = lambda x: positional_encoding(x, num_encoding_functions=num_encoding_functions)
        self.experiment_type = experiment_type
        self.class_embedding_length = class_embedding_length
        self.class_idx = class_idx
        self.experiment_class = experiment_class

    def get_model(self):
        """
        :return: trained deep sdf model loaded from disk
        """
        model = DeepSDFDecoder(self.latent_code_length, self.experiment_type, self.num_encoding_functions, self.class_embedding_length)
        model.load_state_dict(torch.load(self.experiment / "model_best.ckpt", map_location='cpu'))
        model.eval()
        model.to(self.device)
        return model

    def get_latent_codes(self):
        """
        :return: latent codes which were optimized during training
        """
        latent_codes = torch.nn.Embedding.from_pretrained(torch.load(self.experiment / "latent_best.ckpt", map_location='cpu')['weight'])
        train_items = ShapeImplicit(self.experiment_class, 4096, "train", self.experiment_type, num_encoding_functions=2).items
        train_items_class_ids = [int(item.split(' ')[1]) for item in train_items]

        latent_codes.to(self.device)
        return latent_codes, train_items_class_ids

    def get_class_codes(self):
        class_codes = torch.nn.Embedding.from_pretrained(torch.load(self.experiment / "class_best.ckpt", map_location='cpu')['weight'])
        class_codes.to(self.device)
        return class_codes

    def reconstruct(self, points, sdf, num_optimization_iters):
        """
        Reconstructs by optimizing a latent code that best represents the input sdf observations
        :param points: all observed points for the shape which needs to be reconstructed
        :param sdf: all observed sdf values corresponding to the points
        :param num_optimization_iters: optimization is performed for this many number of iterations
        :return: tuple with mesh representations of the reconstruction
        """

        model = self.get_model()

        # TODO: define loss criterion for optimization
        loss_l1 = torch.nn.L1Loss()

        # initialize the latent vector that will be optimized
        latent = torch.ones(1, self.latent_code_length).normal_(mean=0, std=0.01).to(self.device)
        latent.requires_grad = True
        
        # get the class code
        #class_e = torch.ones(1, self.class_embedding_length).normal_(mean=0, std=0.01).to(self.device)
        #class_e.requires_grad = True
        
        if self.experiment_type == 'multi_class':
            class_embedding = self.get_class_codes()(torch.LongTensor([self.class_idx]).to(self.device))
        else:
            class_embedding = None
        #class_embedding = self.get_class_codes()(torch.LongTensor([self.class_idx]).to(self.device))

        # TODO: create optimizer on latent, use a learning rate of 0.005
        optimizer = torch.optim.Adam([latent], lr=0.005)
        
        for iter_idx in range(num_optimization_iters):
            # TODO: zero out gradients
            optimizer.zero_grad()

            # TODO: sample a random batch from the observations, batch size = self.num_samples
            batch_indices = random.sample(range(points.shape[0]), self.num_samples)

            batch_points = points[batch_indices, :] 
            batch_points = self.pe_encoder(batch_points)
            batch_sdf = sdf[batch_indices, :]

            # move batch to device
            batch_points = batch_points.to(self.device)
            batch_sdf = batch_sdf.to(self.device)

            # same latent code is used per point, therefore expand it to have same length as batch points
            latent_codes = latent.expand(self.num_samples, -1)
            if self.experiment_type == 'multi_class':
                class_codes = class_embedding.expand(self.num_samples, -1)
                 # TODO: forward pass with latent_codes and batch_points
                predicted_sdf = model(torch.cat([latent_codes, batch_points, class_codes], dim=1))
            else:
                predicted_sdf = model(torch.cat([latent_codes, batch_points], dim=1))
            
            # TODO: truncate predicted sdf between -0.1, 0.1
            predicted_sdf = torch.clamp(predicted_sdf, -0.1, 0.1)

            # compute loss wrt to observed sdf
            loss = loss_l1(predicted_sdf, batch_sdf)

            # regularize latent code
            loss += 1e-4 * torch.mean(latent.pow(2))

            # TODO: backwards and step
            loss.backward()
            optimizer.step()

            # loss logging
            if iter_idx % 50 == 0:
                print(f'[{iter_idx:05d}] optim_loss: {loss.cpu().item():.6f}')

        print('Optimization complete.')

        print(latent.squeeze(0).shape)
        #print(class_e.shape)
        # visualize the reconstructed shape
        vertices, faces = evaluate_model_on_grid(model, class_embedding, latent.squeeze(0), self.device, 64, None)
        return vertices, faces

    def interpolate(self, latent_code_length, shape_0_id, shape_0_class_id, shape_1_id, shape_1_class_id, num_interpolation_steps):
        """
        Interpolates latent codes between provided shapes and exports the intermediate reconstructions
        :param shape_0_id: first shape identifier
        :param shape_0_class_id: first shape class identifier
        :param shape_1_id: second shape identifier
        :param shape_1_class_id: second shape class identifier
        :param num_interpolation_steps: number of intermediate interpolated points
        :return: None, saves the interpolated shapes to disk
        """

        # get saved model and latent codes
        model = self.get_model()
        train_latent_codes, _ = self.get_latent_codes()
        train_class_codes = self.get_class_codes()
        
        # get indices of shape_ids latent codes
        train_items = ShapeImplicit(self.experiment_class, 4096, "train", experiment_type=self.experiment_type, num_encoding_functions=2).items
        # B084ZBX1YH 1
        
        # Split the first and second items from the train_items list
        train_items_ids = [item.split(' ')[0] for item in train_items]
        
        latent_code_indices = torch.LongTensor([train_items_ids.index(shape_0_id), train_items_ids.index(shape_1_id)]).to(self.device)
        class_code_indices = torch.LongTensor([shape_0_class_id, shape_1_class_id]).to(self.device)
        
        # get latent codes for provided shape
        latent_codes = train_latent_codes(latent_code_indices)
        class_codes = train_class_codes(class_code_indices)
        
        shape_code = torch.cat([latent_codes, class_codes], dim=1)
        
        for i in range(0, num_interpolation_steps + 1):
            # TODO: interpolate the latent codes: latent_codes[0, :] and latent_codes[1, :]
            #interpolated_code = latent_codes[0, :] + (latent_codes[1, :] - latent_codes[0, :]) * i / num_interpolation_steps
            interpolated_latent_code = shape_code[0, :latent_code_length] + (shape_code[1, :latent_code_length] - shape_code[0, :latent_code_length]) * i / num_interpolation_steps
            interpolated_class_code = shape_code[0, latent_code_length:] + (shape_code[1, latent_code_length:] - shape_code[0, latent_code_length:]) * i / num_interpolation_steps
 
            # reconstruct the shape at the interpolated latent code
            evaluate_model_on_grid(model, interpolated_class_code, interpolated_latent_code, self.device, 64, self.experiment / "interpolation" / f"{i:05d}_000.obj")

    
    def interpolate_w_fixed_class_code(self, latent_code_length, shape_0_id, shape_0_class_id, shape_1_id, num_interpolation_steps):
        """
        Interpolates latent codes between provided shapes and exports the intermediate reconstructions
        :param shape_0_id: first shape identifier
        :param shape_0_class_id: first shape class identifier
        :param shape_1_id: second shape identifier
        :param num_interpolation_steps: number of intermediate interpolated points
        :return: None, saves the interpolated shapes to disk
        """

        # get saved model and latent codes
        model = self.get_model()
        train_latent_codes = self.get_latent_codes()
        train_class_codes = self.get_class_codes()
        
        # get indices of shape_ids latent codes
        train_items = ShapeImplicit("multiclass", 4096, "train").items
        # B084ZBX1YH 1
        
        # Split the first and second items from the train_items list
        train_items_ids = [item.split(' ')[0] for item in train_items]
        
        latent_code_indices = torch.LongTensor([train_items_ids.index(shape_0_id), train_items_ids.index(shape_1_id)]).to(self.device)
        class_code_indices = torch.LongTensor([shape_0_class_id]).to(self.device)
        
        # get latent codes for provided shape
        latent_codes = train_latent_codes(latent_code_indices)
        class_codes = train_class_codes(class_code_indices)
        
        shape_code = torch.cat([latent_codes, class_codes], dim=1)
        
        for i in range(0, num_interpolation_steps + 1):
            # TODO: interpolate the latent codes: latent_codes[0, :] and latent_codes[1, :]
            #interpolated_code = latent_codes[0, :] + (latent_codes[1, :] - latent_codes[0, :]) * i / num_interpolation_steps
            interpolated_latent_code = shape_code[0, :latent_code_length] + (shape_code[1, :latent_code_length] - shape_code[0, :latent_code_length]) * i / num_interpolation_steps
            
            # reconstruct the shape at the interpolated latent code
            evaluate_model_on_grid(model, class_codes, interpolated_latent_code, self.device, 64, self.experiment / "interpolation" / f"{i:05d}_000.obj", experiment_type=self.experiment_type)
            
    
    def interpolate_w_random_class_code(self, latent_code_length, shape_0_id, shape_1_id, num_interpolation_steps):
        """
        Interpolates latent codes between provided shapes and exports the intermediate reconstructions
        :param shape_0_id: first shape identifier
        :param shape_1_id: second shape identifier
        :param num_interpolation_steps: number of intermediate interpolated points
        :return: None, saves the interpolated shapes to disk
        """
        model = self.get_model()
        train_latent_codes = self.get_latent_codes()
        
        class_code = torch.rand(2, self.class_embedding_length).normal_(mean=0, std=0.01).to(self.device)
        
        train_items = ShapeImplicit("multiclass", 4096, "train").items
        train_items_ids = [item.split(' ')[0] for item in train_items]
        
        latent_code_indices = torch.LongTensor([train_items_ids.index(shape_0_id), train_items_ids.index(shape_1_id)]).to(self.device)
        
        # get latent codes for provided shape
        latent_codes = train_latent_codes(latent_code_indices)
        print(f'Latent Code: {latent_codes.shape}')
        print(f'Class Code: {class_code.shape}')
        
        shape_code = torch.cat([latent_codes, class_code], dim=1)
        print(f'Shape Code: {shape_code.shape}')
        
        
        for i in range(0, num_interpolation_steps + 1):
            # TODO: interpolate the latent codes: latent_codes[0, :] and latent_codes[1, :]
            #interpolated_code = latent_codes[0, :] + (latent_codes[1, :] - latent_codes[0, :]) * i / num_interpolation_steps
            interpolated_latent_code = shape_code[0, :latent_code_length] + (shape_code[1, :latent_code_length] - shape_code[0, :latent_code_length]) * i / num_interpolation_steps
            print(f'Interpolated Latent Code: {interpolated_latent_code.shape}')
            
            # reconstruct the shape at the interpolated latent code
            evaluate_model_on_grid(model, class_code[0], interpolated_latent_code, self.device, 64, self.experiment / "interpolation" / f"{i:05d}_000.obj", experiment_type=self.experiment_type)
        
    def interpolate_single_class(self, shape_0_id, shape_1_id, num_interpolation_steps):
        model = self.get_model()
        train_latent_codes, _ = self.get_latent_codes()
                
        train_items = ShapeImplicit(self.experiment_class, 4096, "train", experiment_type=self.experiment_class, num_encoding_functions=2).items
        train_items_ids = [item.split(' ')[0] for item in train_items]
        
        latent_code_indices = torch.LongTensor([train_items_ids.index(shape_0_id), train_items_ids.index(shape_1_id)]).to(self.device)
        
        print(f'Latent Code Indices: {latent_code_indices}')
        # get latent codes for provided shape
        latent_codes = train_latent_codes(latent_code_indices)    
        
        for i in range(0, num_interpolation_steps + 1):
            # TODO: interpolate the latent codes: latent_codes[0, :] and latent_codes[1, :]
            interpolated_code = latent_codes[0, :] + (latent_codes[1, :] - latent_codes[0, :]) * i / num_interpolation_steps
            #interpolated_latent_code = shape_code[0, :latent_code_length] + (shape_code[1, :latent_code_length] - shape_code[0, :latent_code_length]) * i / num_interpolation_steps
            
            # reconstruct the shape at the interpolated latent code
            evaluate_model_on_grid(model, None, interpolated_code, self.device, 64, self.experiment / "interpolation" / f"{i:05d}_000.obj")

    def infer_from_latent_code(self, latent_code_index):
        """
        Reconstruct shape from a given latent code index
        :param latent_code_index: shape index for a shape in the train set for which reconstruction is performed
        :return: tuple with mesh representations of the reconstruction
        """

        # get saved model and latent codes
        model = self.get_model()
        train_latent_codes = self.get_latent_codes()
        
        train_class_codes = self.get_class_codes()

        # get latent code at given index
        latent_code_indices = torch.LongTensor([latent_code_index]).to(self.device)
        latent_codes = train_latent_codes(latent_code_indices)
        class_codes = train_class_codes(self.class_idx)

        # reconstruct the shape at latent code
        vertices, faces = evaluate_model_on_grid(model, class_codes, latent_codes[0], self.device, 64, None, self.experiment_type)

        return vertices, faces
    
    def get_number_of_items(self, shape_class, num_points):
        """
        :return: number of items in the train set
        """
        return len(ShapeImplicit(shape_class, num_points, "train").items)