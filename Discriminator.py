import torch
from torch import nn
from Generator import get_noise

class Discriminator(nn.Module):
    def __init__(self, im_dim):
        super(Discriminator, self).__init__()
        # Input will be 64x64x3 = 12288
        self.disc = nn.Sequential(
            self.get_disc_block(im_dim, 512),  # First block: im_dim -> 512
            self.get_disc_block(512, 256),     # Second block: 512 -> 256
            self.get_disc_block(256, 128),     # Third block: 256 -> 128
            self.get_disc_block(128, 64),      # Fourth block: 128 -> 64
            nn.Linear(64, 1)                   # Final layer: 64 -> 1 (output scalar score)
        )

    def get_disc_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, image):
        return self.disc(image)

def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    '''
    Function to calculate the discriminator's loss.
    '''
    noise = get_noise(num_images, z_dim, device=device)
    fake_images = gen(noise)

    # Get predictions for real and fake images
    disc_real_pred = disc(real)
    disc_fake_pred = disc(fake_images.detach())

    # Calculate the loss for both real and fake images
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_loss = (disc_real_loss + disc_fake_loss) / 2
    return disc_loss
