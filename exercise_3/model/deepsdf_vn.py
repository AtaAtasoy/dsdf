import torch.nn as nn
import torch

from exercise_3.model.vnn import VNN_ResnetPointnet

class DeepSDFVNDecoder(nn.Module):

    def __init__(self, latent_size, experiment_type='vanilla', num_encoding_functions=4):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2
        self.num_encoding_functions = num_encoding_functions

        self.encoder_output_dim = 256
        self.encoder = VNN_ResnetPointnet(c_dim=self.encoder_output_dim, hidden_dim=256, experiment_type=experiment_type, num_encoding_functions=num_encoding_functions)
        # TODO: Define model
        '''
        if experiment_type == 'pe':
            self.lin0 = torch.nn.utils.weight_norm(nn.Linear(latent_size +  3 + 3 * 2 * num_encoding_functions, 512))
        else: 
            self.lin0 = torch.nn.utils.weight_norm(nn.Linear(latent_size + 3, 512))
        '''
        # TODO: Test
        self.lin0 = torch.nn.utils.weight_norm(nn.Linear(latent_size + self.encoder_output_dim + 3, 512))
        self.relu0 = nn.ReLU()
        self.drop0 = nn.Dropout(dropout_prob)

        self.lin1 = torch.nn.utils.weight_norm(nn.Linear(512, 512))
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_prob)

        self.lin2 = torch.nn.utils.weight_norm(nn.Linear(512, 512))
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_prob)

        self.lin3 = torch.nn.utils.weight_norm(nn.Linear(512, 512))
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout_prob)

        '''
        if experiment_type == 'pe':
            self.lin4 = torch.nn.utils.weight_norm(nn.Linear(512, 512 - (latent_size + 3 + 3 * 2 * num_encoding_functions)))
        else:
            self.lin4 = torch.nn.utils.weight_norm(nn.Linear(512, 512 - (latent_size + 3)))
        '''
        
        self.lin4 =  torch.nn.utils.weight_norm(nn.Linear(512, 512))
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(dropout_prob)

        self.lin5 = torch.nn.utils.weight_norm(nn.Linear(512 + (latent_size + self.encoder_output_dim + 3), 512))
        self.relu5 = nn.ReLU()
        self.drop5 = nn.Dropout(dropout_prob)

        self.lin6 = torch.nn.utils.weight_norm(nn.Linear(512, 512))
        self.relu6 = nn.ReLU()
        self.drop6 = nn.Dropout(dropout_prob)

        self.lin7 = torch.nn.utils.weight_norm(nn.Linear(512, 512))
        self.relu7 = nn.ReLU()
        self.drop7 = nn.Dropout(dropout_prob)

        self.lin8 = torch.nn.utils.weight_norm(nn.Linear(512, 1))

    def forward(self, x_in, latent_code):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        # TODO: implement forward pass
        x = torch.unsqueeze(x_in, 0)
        x = self.encoder(x)
        x = torch.squeeze(x, 0)
        x = torch.squeeze(x, 1)
        x = x.transpose(0, 1)
        # Concat the encoded input and the latent_code in the latent space
        x = torch.cat((x, latent_code), dim=1)
        x = torch.cat((x, x_in), dim=1)
        x_input = x # initial input of DeepSDF
        x = self.lin0(x) 
        x = self.relu0(x)
        x = self.drop0(x)

        x = self.lin1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.lin2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.lin3(x)
        x = self.relu3(x)
        x = self.drop3(x)

        x = self.lin4(x)
        x = self.relu4(x)
        x = self.drop4(x)

        # Skip connection
        x = torch.cat((x, x_input), dim=1)
        x = self.lin5(x)
        x = self.relu5(x)

        x = self.lin6(x)
        x = self.relu6(x)
        x = self.drop6(x)

        x = self.lin7(x)
        x = self.relu7(x)
        x = self.drop7(x)

        x = self.lin8(x)

        return x