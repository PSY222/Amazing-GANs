import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as plt
import numpy as np
import torchvision.utils as utils
from scipy.stats import truncnorm


def show_tensor_images(image_tensor, num_images=16, size=(3, 64, 64), nrow=3):
    image_tensor = (image_tensor + 1) / 2   #normalize the image
    image_unflat = image_tensor.detach().cpu().clamp_(0, 1)
    image_grid = utils.make_grid(image_unflat[:num_images], nrow=nrow, padding=0)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.show()

#Truncation Trick
def get_truncated_noise(n_samples,z_dim,truncation):
    truncated_noise = truncnorm.rvs(-truncation,truncation,size=(n_samples,z_dim))
    return torch.Tensor(truncated_noise)


#Mapping  𝑧  →  𝑤
#w_dim: intermediate noise vector dimension
#z_dim: the dimension of noise vector
class MappingLayers(nn.Module):
    def __init__(self,z_dim,hidden_dim,w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(z_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,w_dim)
        )

    def forward(self,noise):
        return self.mapping(noise)
    
    def get_mapping(self):
        return self.mapping
    
#Random noise injection
class InjectNoise(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1,channels,1,1))

    def forward(self,image):
        noise_shape = (image.shape[0],1, image.shape[2],image.shape[3])
        noise = torch.randn(noise_shape,device=image.device)
        return image +self.weight * noise
    
    
#Adaptive Instance Normalization(AdaIN)
class AdaIN(nn.Module):
    def __init__(self,channels,w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale_transform = nn.Linear(w_dim, channels)
        self.style_shift_transform = nn.Linear(w_dim, channels)
    
    def forward(self, image, w):
        normalized_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]
        transformed_image = style_scale * normalized_image + style_shift
        return transformed_image
    
    def get_style_scale_transform(self):
        return self.style_scale_transform
    
    def get_style_shift_transform(self):
        return self.style_shift_transform
    

class MicroStyleGANGeneratorBlock(nn.Module):
    def __init__(self, in_chan, out_chan, w_dim, kernel_size, starting_size, use_upsample=True):
        super().__init__()
        self.use_upsample = use_upsample

        if self.use_upsample:
            self.upsample = nn.Upsample((starting_size,starting_size), mode='bilinear')
        self.conv = nn.Conv2d(in_chan,out_chan, kernel_size, padding=1) 
        self.inject_noise = InjectNoise(out_chan)
        self.adain = AdaIN(out_chan, w_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        if self.use_upsample:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.inject_noise(x)
        x = self.adain(x, w)
        x = self.activation(x)
        return x
    
class MicroStyleGANGenerator(nn.Module):
    def __init__(self,z_dim,map_hidden_dim,w_dim,in_chan,out_chan,kernel_size,hidden_chan):
        super().__init__()
        self.map = MappingLayers(z_dim, map_hidden_dim, w_dim)
        # Typically this constant is initiated to all ones, but you will initiate to a
        # Gaussian to better visualize the network's effect
        self.starting_constant = nn.Parameter(torch.randn(1, in_chan, 4, 4))
        self.block0 = MicroStyleGANGeneratorBlock(in_chan, hidden_chan, w_dim, kernel_size, 4, use_upsample=False)
        self.block1 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 8)
        self.block2 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 16)
        # You need to have a way of mapping from the output noise to an image, 
        # so you learn a 1x1 convolution to transform the e.g. 512 channels into 3 channels
        # (Note that this is simplified, with clipping used in the real StyleGAN)
        self.block1_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.block2_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.alpha = 0.2

    def upsample_to_match_size(self, smaller_image, bigger_image):
        return F.interpolate(smaller_image, size=bigger_image.shape[-2:], mode='bilinear')

    def forward(self, noise, return_intermediate=False):
        x = self.starting_constant
        w = self.map(noise)
        x = self.block0(x, w)
        x_small = self.block1(x, w) # First generator run output
        x_small_image = self.block1_to_image(x_small)
        x_big = self.block2(x_small, w) # Second generator run output 
        x_big_image = self.block2_to_image(x_big)
        x_small_upsample = self.upsample_to_match_size(x_small_image, x_big_image) # Upsample first generator run output to be same size as second generator run output 
        # Interpolate between the upsampled image and the image from the generator using alpha
       
        interpolation = self.alpha*(x_big_image) + (1-self.alpha)*(x_small_upsample)
        
        if return_intermediate:
            return interpolation, x_small_upsample, x_big_image
        return interpolation
    
#Visualize the results
viz_samples = 10
z_dim = 128
out_chan = 3
truncation = 0.7
# The noise is exaggerated for visual effect
viz_noise = get_truncated_noise(viz_samples, z_dim, truncation) * 10
mu_stylegan = MicroStyleGANGenerator(z_dim=z_dim,map_hidden_dim=1024,w_dim=496, in_chan=512,out_chan=out_chan,kernel_size=3,hidden_chan=256)

mu_stylegan.eval()
images = []
for alpha in np.linspace(0, 1, num=5):
    mu_stylegan.alpha = alpha
    viz_result, _, _ =  mu_stylegan(viz_noise,return_intermediate=True)
    images += [tensor for tensor in viz_result]
show_tensor_images(torch.stack(images), nrow=viz_samples, num_images=len(images))
mu_stylegan = mu_stylegan.train()
    
  