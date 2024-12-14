class Generator3D(nn.Module):
    def __init__(self, latent_dim=100, base_channels=64):
        super(Generator3D, self).__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, base_channels*8, 2, 1, 0, bias=False),
            nn.BatchNorm3d(base_channels*8),
            nn.ReLU(True),

            nn.ConvTranspose3d(base_channels*8, base_channels*4, 2, 2, 0, bias=False),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(True),

            nn.ConvTranspose3d(base_channels*4, base_channels*2, 2, 2, 0, bias=False),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(True),

            nn.ConvTranspose3d(base_channels*2, base_channels, 2, 2, 0, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(True),

            nn.Conv3d(base_channels, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = z.view(z.size(0), self.latent_dim, 1, 1, 1)
        return self.net(z)

class Discriminator3D(nn.Module):
    def __init__(self, base_channels=64):
        super(Discriminator3D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(base_channels, base_channels*2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(base_channels*2, base_channels*4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(base_channels*4, base_channels*8, 2, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(base_channels*8, 1, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        out = self.net(x).view(-1)  # shape (N,)
        return out
