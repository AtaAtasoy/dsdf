import torch.nn as nn
import torch
from exercise_3.model.vn_layers import *


class DeepSDFVNDecoder(nn.Module):

    def __init__(self, latent_size, num_encoding_functions=6):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2
    
        # TODO: Define model
        self.lin0 = VNLinearAndLeakyReLU(latent_size + 3 + 3 * 2 * num_encoding_functions, 512//3, use_batchnorm='none', negative_slope=0.0)
        self.drop0 = nn.Dropout(dropout_prob)

        self.lin1 = VNLinearAndLeakyReLU(512//3, 512//3, use_batchnorm='none', negative_slope=0.0)
        self.drop1 = nn.Dropout(dropout_prob)

        self.lin2 = VNLinearAndLeakyReLU(512//3, 512//3, use_batchnorm='none', negative_slope=0.0)
        self.drop2 = nn.Dropout(dropout_prob)

        self.lin3 = VNLinearAndLeakyReLU(512//3, 512//3, use_batchnorm='none', negative_slope=0.0)
        self.drop3 = nn.Dropout(dropout_prob)

        self.lin4 = VNLinearAndLeakyReLU(512//3, (512 - (latent_size + 3 + 3 * 2 * num_encoding_functions)), use_batchnorm='none', negative_slope=0.0)
        self.drop4 = nn.Dropout(dropout_prob)

        self.lin5 = VNLinearAndLeakyReLU(512, 512//3, use_batchnorm='none', negative_slope=0.0)
        self.drop5 = nn.Dropout(dropout_prob)

        self.lin6 = VNLinearAndLeakyReLU(512//3, 512//3, use_batchnorm='none', negative_slope=0.0)
        self.drop6 = nn.Dropout(dropout_prob)

        self.lin7 = VNLinearAndLeakyReLU(512//3, 512//3, use_batchnorm='none', negative_slope=0.0)
        self.drop7 = nn.Dropout(dropout_prob)

        self.lin8 = torch.nn.utils.weight_norm(nn.Linear(512//3, 1))

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        B, D, N = x_in.shape
        # TODO: implement forward pass
        x = self.lin0(x_in)
        x = self.drop0(x)

        x = self.lin1(x)
        x = self.drop1(x)

        x = self.lin2(x)
        x = self.drop2(x)

        x = self.lin3(x)
        x = self.drop3(x)

        x = self.lin4(x)
        x = self.drop4(x)

        # Skip connection
        x = torch.cat((x, x_in), dim=1)
        x = self.lin5(x)

        x = self.lin6(x)
        x = self.drop6(x)

        x = self.lin7(x)
        x = self.drop7(x)

        x = x.transpose(1, -1)
        x = self.lin8(x)

        return x
