import torch
import torch.nn as nn


class ThreeDEPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_features = 80

        # TODO: 4 Encoder layers
        self.e1 = nn.Sequential(
            nn.Conv3d(2, self.num_features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.e2 = nn.Sequential(
            nn.Conv3d(self.num_features, self.num_features * 2, 4, stride=2, padding=1),
            nn.BatchNorm3d(self.num_features * 2),
            nn.LeakyReLU(0.2)
        )

        self.e3 = nn.Sequential(
            nn.Conv3d(self.num_features * 2, self.num_features * 4, 4, stride=2, padding=1),
            nn.BatchNorm3d(self.num_features * 4),
            nn.LeakyReLU(0.2)
        )

        self.e4 = nn.Sequential(
            nn.Conv3d(self.num_features * 4, self.num_features * 8, 4, stride=1, padding=0),
            nn.BatchNorm3d(self.num_features * 8),
            nn.LeakyReLU(0.2)
        )

        # TODO: 2 Bottleneck layers
        self.bottleneck = nn.Sequential(
            nn.Linear(self.num_features * 8, self.num_features * 8),
            nn.ReLU(),
            nn.Linear(self.num_features * 8, self.num_features * 8),
            nn.ReLU()
        )

        # TODO: 4 Decoder layers
        self.d1 = nn.Sequential(
            nn.ConvTranspose3d(self.num_features * 8 * 2, self.num_features * 4, 4, stride=1, padding=0),
            nn.BatchNorm3d(self.num_features * 4),
            nn.LeakyReLU(0.2)
        )

        self.d2 = nn.Sequential(
            nn.ConvTranspose3d(self.num_features * 4 * 2, self.num_features * 2, 4, stride=2, padding=1),
            nn.BatchNorm3d(self.num_features * 2),
            nn.LeakyReLU(0.2)
        )

        self.d3 = nn.Sequential(
            nn.ConvTranspose3d(self.num_features * 2 * 2, self.num_features, 4, stride=2, padding=1),
            nn.BatchNorm3d(self.num_features),
            nn.LeakyReLU(0.2)
        )

        self.d4 = nn.ConvTranspose3d(self.num_features * 2, 1, 4, stride=2, padding=1)

    def forward(self, x):
        b = x.shape[0]
        # Encode
        # TODO: Pass x though encoder while keeping the intermediate outputs for the skip connections
        x_e1 = self.e1(x)
        #print("x_e1_shape", x_e1.shape)
        x_e2 = self.e2(x_e1)
        #print("x_e2_shape", x_e2.shape)
        x_e3 = self.e3(x_e2)
        #print("x_e3_shape", x_e3.shape)
        x_e4 = self.e4(x_e3)
        #print("x_e4_shape", x_e4.shape)

        # Reshape and apply bottleneck layers
        x = x_e4.view(b, -1)
        x = self.bottleneck(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)
        #print("Bottleneck output:", x.shape)

        # Decode
        # TODO: Pass x through the decoder, applying the skip connections in the process
        x = self.d1(torch.cat((x, x_e4), dim=1))
        #print("Decoder output of first layer:", x.shape)
        x = self.d2(torch.cat((x, x_e3), dim=1))
        #print("Decoder output of second layer:", x.shape)
        x = self.d3(torch.cat((x, x_e2), dim=1))
        #print("Decoder output of third layer:", x.shape)
        x = self.d4(torch.cat((x, x_e1), dim=1))
        #print("Decoder output of fourth layer:", x.shape)

        x = torch.squeeze(x, dim=1)
        # TODO: Log scaling
        x = torch.log(torch.abs(x) + 1)

        return x
