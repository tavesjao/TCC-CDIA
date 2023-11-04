import torch
from torch import nn
from torchvision import datasets, transforms
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
    
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
    
class GAN():
    def __init__(self, latent_dim, img_shape):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.generator = Generator(self.latent_dim, self.img_shape)
        self.discriminator = Discriminator(self.img_shape)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        
    def train(self, n_epochs, batch_size, sample_interval=100):
        dataloader = self.load_data(batch_size)
        for epoch in range(n_epochs):
            for i, (imgs, _) in enumerate(dataloader):
                self.train_discriminator(imgs)
                self.train_generator(batch_size)
                if i % sample_interval == 0:
                    print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {self.loss_D.item():.6f}] [G loss: {self.loss_G.item():.6f}]")
                    self.sample_images(epoch, i)
                    
    def train_discriminator(self, imgs):
        self.discriminator.zero_grad()
        real_imgs = imgs.to(self.device)
        valid = torch.ones(real_imgs.size(0), 1).to(self.device)
        fake = torch.zeros(real_imgs.size(0), 1).to(self.device)
        real_loss = self.criterion(self.discriminator(real_imgs), valid)
        z = torch.randn(real_imgs.size(0), self.latent_dim).to(self.device)
        fake_imgs = self.generator(z)
        fake_loss = self.criterion(self.discriminator(fake_imgs.detach()), fake)
        self.loss_D = (real_loss + fake_loss) / 2
        self.loss_D.backward()
        self.optimizer_D.step()
        
    def train_generator(self, batch_size):
        self.generator.zero_grad()
        valid = torch.ones(batch_size, 1).to(self.device)
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_imgs = self.generator(z)
        self.loss_G = self.criterion(self.discriminator(fake_imgs), valid)
        self.loss_G.backward()
        self.optimizer_G.step()

    def sample_images(self, epoch, i):
        os.makedirs('images', exist_ok=True)
        z = torch.randn(5, self.latent_dim).to(self.device)
        gen_imgs = self.generator(z)
        torchvision.utils.save_image(gen_imgs.data, f"images/{epoch}_{i}.png", nrow=5, normalize=True)

    def load_data(self, batch_size):
        os.makedirs('../data/processed/train/', exist_ok=True)
        dataset = datasets.ImageFolder(root='../data/frames', transform=transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader
    
    def save_model(self):
        torch.save(self.generator.state_dict(), '../models/generator.pth')
        torch.save(self.discriminator.state_dict(), '../models/discriminator.pth')