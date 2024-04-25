import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os
import torchvision.utils as vutils
import numpy as np

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3):
        super().__init__()
        self.gen = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 32 x 32
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. img_channels x 64 x 64
        )

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        self.dis = nn.Sequential(
            # input is (img_channels) x 64 x 64
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 512 x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.dis(x)

class DCGAN():
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        print(f'Using device: {self.device}')
        self.netG = Generator().to(device)  # Define Generator
        self.netD = Discriminator().to(device)  # Define Discriminator
        self.optG = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optD = optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.fixed_noise = torch.randn(64, 100, 1, 1, device=self.device)  # Fixed noise for observing generator progress

        # Initialize weights
        self.netG.apply(self.weights_init)
        self.netD.apply(self.weights_init)

    def train(self, epochs):
        criterion = nn.BCELoss()
        G_losses = []
        D_losses = []
        img_list = []

        for epoch in range(epochs):
            for i, (images, _) in enumerate(self.dataloader):
                # Training Discriminator
                self.netD.zero_grad()
                real_images = images.to(self.device)
                batch_size = real_images.size(0)
                labels = torch.full((batch_size,), 1., dtype=torch.float, device=self.device)

                output = self.netD(real_images).view(-1)
                errD_real = criterion(output, labels)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(batch_size, 100, 1, 1, device=self.device)
                fake_images = self.netG(noise)
                labels.fill_(0.)
                output = self.netD(fake_images.detach()).view(-1)
                errD_fake = criterion(output, labels)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                self.optD.step()
                torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1)

                # Training Generator
                self.netG.zero_grad()
                labels.fill_(1.)  # fake labels are real for generator cost
                output = self.netD(fake_images).view(-1)
                errG = criterion(output, labels)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optG.step()
                torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=1)

                # Output training stats
                if i % 10 == 0:
                    print(f'[{epoch}/{epochs}][{i}/{len(self.dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')

                # Check how the generator is doing by saving G's output on fixed_noise
                if (epoch % 10 == 0) and (i == len(self.dataloader)-1):
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    # Optionally, show or save the image
                    plt.figure(figsize=(8,8))
                    plt.axis("off")
                    plt.title("Generated Images")
                    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
                    plt.show()

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

            # Save models periodically
            if epoch % 10 == 0:
                torch.save(self.netG.state_dict(), f'../models/dcGan/generator_epoch_{epoch}.pth')
                torch.save(self.netD.state_dict(), f'../models/dcGan/discriminator_epoch_{epoch}.pth')

        torch.save(self.netG.state_dict(), '../models/dcGan/generator_final.pth')
        torch.save(self.netD.state_dict(), '../models/dcGan/discriminator_final.pth')
            

        print('Training finished.')

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)