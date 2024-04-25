import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False)  # No sigmoid activation
        )

    def forward(self, x):
        return self.dis(x).view(-1)

class WGAN():
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        print(f'Using device: {self.device}')
        self.netG = Generator().to(device)  # Initialize Generator
        self.netD = Discriminator().to(device)  # Initialize Discriminator
        self.optG = optim.RMSprop(self.netG.parameters(), lr=0.00005)
        self.optD = optim.RMSprop(self.netD.parameters(), lr=0.00005)
        
        self.fixed_noise = torch.randn(64, 100, 1, 1, device=self.device)  # Fixed noise for observing generator progress

        # Initialize weights
        self.netG.apply(self.weights_init)
        self.netD.apply(self.weights_init)

    def train(self, epochs):
        G_losses = []
        D_losses = []
        img_list = []

        for epoch in range(epochs):
            for i, (images, _) in enumerate(self.dataloader):
                # Training Discriminator
                self.netD.zero_grad()
                real_images = images.to(self.device)
                batch_size = real_images.size(0)
                real_output = self.netD(real_images)
                real_loss = -torch.mean(real_output)

                noise = torch.randn(batch_size, 100, 1, 1, device=self.device)
                fake_images = self.netG(noise)
                fake_output = self.netD(fake_images.detach())
                fake_loss = torch.mean(fake_output)

                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.optD.step()

                # Apply weight clipping
                for p in self.netD.parameters():
                    p.data.clamp_(-0.01, 0.01)

                # Train Generator every 5 discriminator iterations
                if i % 5 == 0:
                    self.netG.zero_grad()
                    output = self.netD(fake_images)
                    g_loss = -torch.mean(output)
                    g_loss.backward()
                    self.optG.step()

                # Output training stats and save images
                if i % 50 == 0:
                    print(f'[{epoch}/{epochs}][{i}/{len(self.dataloader)}] Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f}')
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                        plt.figure(figsize=(8,8))
                        plt.axis("off")
                        plt.title("Generated Images")
                        plt.imshow(np.transpose(img_list[-1], (1,2,0)))
                        plt.show()

                # Save Losses for plotting later
                G_losses.append(g_loss.item())
                D_losses.append(d_loss.item())

            # Save models periodically
            if epoch % 10 == 0:
                torch.save(self.netG.state_dict(), f'../models/wGan/generator_epoch_{epoch}.pth')
                torch.save(self.netD.state_dict(), f'../models/wGan/discriminator_epoch_{epoch}.pth')

        torch.save(self.netG.state_dict(), '../models/wGan/generator_final.pth')
        torch.save(self.netD.state_dict(), '../models/wGan/discriminator_final.pth')
        print('Training finished.')

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
