import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(0)  # For reproducibility

from Generator import Generator, get_noise, get_gen_loss
from Discriminator import Discriminator, get_disc_loss

# ImageNet Mini uses 64x64 images and 3 color channels (RGB)
im_size = 64
channels = 3
im_dim = im_size * im_size * channels

def show_tensor_images(gen_images, real_images, num_images=25, size=(channels, im_size, im_size)):
    '''
    Function for visualizing images: Given two tensors of images (generated and real),
    number of images to display, and size per image, plots and prints the images in a uniform grid.
    '''
    gen_images_unflat = gen_images.detach().cpu().view(-1, *size)
    real_images_unflat = real_images.detach().cpu().view(-1, *size)
    
    # Concatenate the generated and real images along the width
    images_combined = torch.cat([gen_images_unflat[:num_images], real_images_unflat[:num_images]], dim=0)
    
    image_grid = torchvision.utils.make_grid(images_combined, nrow=num_images, padding=2, normalize=True)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.show()

# Set your parameters
criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.00001
device = 'cpu'

# Transformations for the ImageNet Mini dataset (resize and normalization)
transform = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Load ImageNet Mini dataset
dataloader = DataLoader(
    datasets.ImageFolder('./all_images/', transform=transform),
    batch_size=batch_size,
    shuffle=True
)

# Initialize generator and discriminator
gen = Generator(z_dim=z_dim, im_dim=im_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator(im_dim=im_dim).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True
gen_loss = False
error = False

for epoch in range(n_epochs):
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)

        # Flatten the real images from (B, C, H, W) to (B, im_dim)
        real = real.view(cur_batch_size, -1).to(device)

        ### Update discriminator ###
        disc_opt.zero_grad()
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
        disc_loss.backward()
        disc_opt.step()

        if test_generator:
            old_generator_weights = gen.gen[0][0].weight.detach().clone()

        ### Update generator ###
        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        gen_loss.backward()
        gen_opt.step()

        if test_generator:
            try:
                assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
            except:
                error = True
                print("Runtime tests have failed")

        mean_discriminator_loss += disc_loss.item() / display_step
        mean_generator_loss += gen_loss.item() / display_step

        ### Visualization code ###
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            show_tensor_images(fake, real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1
