import torch.nn as nn

KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
A = 8
B = 16
C = 32
D = 64
E = 128
F = 256
G = 512
H = 1024

LS = 256

# Define the Autoencoder class
class Encoder(nn.Module):
    # input shape in tensor form: (1, 256, 256), i.e., patches of size 256x256

    def __init__(self, output_dim=LS):
        super(Encoder, self).__init__()

        self.output_dim = output_dim

        # Encoder
        self.encoder = nn.Sequential(

            nn.Conv2d(1, A, KERNEL_SIZE, STRIDE, PADDING),
            nn.BatchNorm2d(A),
            nn.ReLU(),
            nn.Conv2d(A, B, KERNEL_SIZE, STRIDE, PADDING),
            nn.BatchNorm2d(B),
            nn.ReLU(),
            nn.Conv2d(B, C, KERNEL_SIZE, STRIDE, PADDING),
            nn.BatchNorm2d(C),
            nn.ReLU(),
            nn.Conv2d(C, D, KERNEL_SIZE, STRIDE, PADDING),
            nn.BatchNorm2d(D),
            nn.ReLU(),
            nn.Conv2d(D, E, KERNEL_SIZE, STRIDE, PADDING),
            nn.BatchNorm2d(E),
            nn.ReLU(),
            nn.Conv2d(E, F, KERNEL_SIZE, STRIDE, PADDING),
            nn.BatchNorm2d(F),
            nn.ReLU(),
            nn.Conv2d(F, G, KERNEL_SIZE, STRIDE, PADDING),
            nn.BatchNorm2d(G),
            nn.ReLU(),
            nn.Conv2d(G, H, KERNEL_SIZE, STRIDE, PADDING),
            nn.BatchNorm2d(H),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(H, LS)
        )

    def forward(self, x):
        x = x.float()
        x = self.encoder(x)
        return x

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


class Decoder(nn.Module):

    def __init__(self, input_dim=LS):
        super(Decoder, self).__init__()

        self.input_dim = input_dim

        self.linear1 = nn.Linear(LS, H)
        self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(F, G)


        # Decoder
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(H, G, KERNEL_SIZE, STRIDE, PADDING, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(G, F, KERNEL_SIZE, STRIDE, PADDING, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(F, E, KERNEL_SIZE, STRIDE, PADDING, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(E, D, KERNEL_SIZE, STRIDE, PADDING, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(D, C, KERNEL_SIZE, STRIDE, PADDING, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(C, B, KERNEL_SIZE, STRIDE, PADDING, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(B, A, KERNEL_SIZE, STRIDE, PADDING, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(A, 1, KERNEL_SIZE, STRIDE, PADDING, output_padding=1),
            nn.Sigmoid()  # ensures that the output values are in range (0,1)
        )
        # self.output = nn.Conv2d(1, 1, kernel_size=1, stride=1)  # no tenc clar que fa aixo

    def forward(self, x):
        x = x.float()
        x = self.linear1(x)
        # x = self.relu(x)
        # x = self.linear2(x)
        # reshape 3d tensor to 4d tensor
        x = x.reshape(x.shape[0], H, 1, 1)
        x = self.decoder(x)
        # return self.output(x)
        return x


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


# putting them together:
class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(output_dim=LS)
        self.decoder = Decoder(input_dim=LS)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def instance():
    return AutoEncoder()
