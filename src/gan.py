import torch
from torch import nn
from torch.nn import (
    Module, 
    Sequential, 
    ConvTranspose2d, 
    BatchNorm2d, 
    ReLU, 
    Tanh, 
    Conv2d, 
    LeakyReLU, 
    Sigmoid)
import numpy as np
import os
import matplotlib.pyplot as plt

class Generator(Module):
    def __init__(self):
 
        # calling constructor of parent class
        super().__init__()
 
        self.gen = Sequential(
          ConvTranspose2d(in_channels = 100, out_channels = 512 , kernel_size = 4, stride = 1, padding = 0, bias = False),
          # the output from the above will be b_size ,512, 4,4
          BatchNorm2d(num_features = 512), # From an input of size (b_size, C, H, W), pick num_features = C
          ReLU(inplace = True),
 
          ConvTranspose2d(in_channels = 512, out_channels = 256 , kernel_size = 4, stride = 2, padding = 1, bias = False),
          # the output from the above will be b_size ,256, 8,8
          BatchNorm2d(num_features = 256),
          ReLU(inplace = True),
 
          ConvTranspose2d(in_channels = 256, out_channels = 128 , kernel_size = 4, stride = 2, padding = 1, bias = False),
          # the output from the above will be b_size ,128, 16,16
          BatchNorm2d(num_features = 128),
          ReLU(inplace = True),
 
          ConvTranspose2d(in_channels = 128, out_channels = 3 , kernel_size = 4, stride = 2, padding = 1, bias = False),
          # the output from the above will be b_size ,3, 32,32
          Tanh()
         
        )
 
    def forward(self, input):
        return self.gen(input)
    
class Discriminator(Module):
    def __init__(self):
 
        super().__init__()
        self.dis = Sequential(
 
            # input is (3, 32, 32)
            Conv2d(in_channels = 3, out_channels = 32, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 32, 16, 16
            LeakyReLU(0.2, inplace=True),
 
            Conv2d(in_channels = 32, out_channels = 32*2, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 32*2, 8, 8
            BatchNorm2d(32 * 2),
            LeakyReLU(0.2, inplace=True),
 
            Conv2d(in_channels = 32*2, out_channels = 32*4, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 32*4, 4, 4
            BatchNorm2d(32 * 4),
            LeakyReLU(0.2, inplace=True),
 
            Conv2d(in_channels = 32*4, out_channels = 32*8, kernel_size = 4, stride = 2, padding = 1, bias=False),
            # ouput from above layer is b_size, 256, 2, 2
            # NOTE: spatial size of this layer is 2x2, hence in the final layer, the kernel size must be 2 instead (or smaller than) 4
            BatchNorm2d(32 * 8),
            LeakyReLU(0.2, inplace=True),
 
            Conv2d(in_channels = 32*8, out_channels = 1, kernel_size = 2, stride = 2, padding = 0, bias=False),
            # ouput from above layer is b_size, 1, 1, 1
            Sigmoid()
        )
     
    def forward(self, input):
        return self.dis(input)

class GAN:
    def __init__(self, dataloader, device, netD, netG, opt_D, opt_G, loss, plot_images):
        self.dataloader = dataloader
        self.device = device
        self.netD = netD
        self.netG = netG
        self.opt_D = opt_D
        self.opt_G = opt_G
        self.loss = loss
        self.plot_images = plot_images
        self.d_losses = []
        self.g_losses = []

    def train(self, epochs):
        print_interval = 100
        for e in range(epochs):
            for i, b in enumerate(self.dataloader):
                d_loss = self.update_discriminator(b)
                g_loss = self.update_generator(b)
                self.d_losses.append(d_loss.item())
                self.g_losses.append(g_loss.item())
                if e % print_interval == 0 and i == 0:
                    self.plot_generator_images(b)

    def update_discriminator(self, b):
        self.opt_D.zero_grad()
        yhat = self.netD(b.to(self.device)).view(-1)
        target = torch.ones(len(b), dtype=torch.float, device=self.device)
        loss_real = self.loss(yhat, target)
        loss_real.backward()

        noise = torch.randn(len(b), 100, 1, 1, device=self.device)
        fake_img = self.netG(noise)
        yhat = self.netD(fake_img.detach()).view(-1)
        target = torch.zeros(len(b), dtype=torch.float, device=self.device)
        loss_fake = self.loss(yhat, target)
        loss_fake.backward()

        self.opt_D.step()
        return loss_real + loss_fake

    def update_generator(self, b):
        self.opt_G.zero_grad()
        noise = torch.randn(len(b), 100, 1, 1, device=self.device)
        fake_img = self.netG(noise)
        yhat = self.netD(fake_img).view(-1)
        target = torch.ones(len(b), dtype=torch.float, device=self.device)
        loss_gen = self.loss(yhat, target)
        loss_gen.backward()
        self.opt_G.step()
        return loss_gen

    def plot_generator_images(self, b):
        noise = torch.randn(len(b), 100, 1, 1, device=self.device)
        fake_img = self.netG(noise)
        img_plot = np.transpose(fake_img.detach().cpu(), (0,2,3,1))
        self.plot_images(img_plot)