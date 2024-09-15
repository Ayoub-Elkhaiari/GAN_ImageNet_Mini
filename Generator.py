import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, z_dim, im_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Output should be 64x64x3 = 12288
        self.gen = nn.Sequential(
            self.get_gen_block(z_dim, 512),  # First block: z_dim -> 512
            self.get_gen_block(512, 256),    # Second block: 512 -> 256
            self.get_gen_block(256, 128),    # Third block: 256 -> 128
            self.get_gen_block(128, 64),     # Fourth block: 128 -> 64
            nn.Linear(64, im_dim),           # Final layer: 64 -> im_dim (64x64x3)
            nn.Tanh()                        # Output: scaled between -1 and 1
        )

    def get_gen_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, noise):
        return self.gen(noise)

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)

def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    '''
    Function to calculate the generator's loss.
    '''
    noise = get_noise(num_images, z_dim, device=device)
    fake_images = gen(noise)
    disc_fake_pred = disc(fake_images)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss
