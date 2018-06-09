import argparse

import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from scripts.model import Discriminator, Generator, noise, init_weights
from scripts.helper_cifar import ones_tensor, zeros_tensor, to_image_tensor
from scripts.data import get_data
from visualization.plotter import TensorboardPlotter

parser = argparse.ArgumentParser(description='Train CGAN for CIFAR')
parser.add_argument('-b', '--batch-size', type=int, default=100)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0002)
parser.add_argument('-n', '--num_epochs', type=int, default=100)
parser.add_argument('-log', '--log_interval', type=int, default=1)
parser.add_argument('-title', '--title', type=str, default='CGAN for CIFAR10')
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('-b1', '--beta1', type=float, default=0.5)
parser.add_argument('-b2', '--beta2', type=float, default=0.999)

args = parser.parse_args()

batch_size = args.batch_size
learning_rate = args.learning_rate
num_epochs = args.num_epochs
log_interval = args.log_interval
title = args.title
image_size = args.image_size
beta1 = args.beta1
beta2 =args.beta2

plotter = TensorboardPlotter(title)

data = get_data(image_size)

dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

discriminator = Discriminator()
discriminator.apply(init_weights)
generator = Generator()
generator.apply(init_weights)

if torch.cuda.is_available():
    discriminator.cuda()
    generator.cuda()

d_optim = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))
g_optim = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))

loss_function = nn.BCELoss()


def discriminator_step(optimizer, loss_function, true_data, fake_data):
    optimizer.zero_grad()
    output_true = discriminator(true_data)
    loss_true = loss_function(output_true, ones_tensor(true_data.size(0)))
    loss_true.backward()

    output_fake = discriminator(fake_data)
    loss_fake = loss_function(output_fake, zeros_tensor(fake_data.size(0)))
    loss_fake.backward()

    optimizer.step()
    return loss_true + loss_fake


def generator_step(optimizer, loss_function, fake_data):
    optimizer.zero_grad()
    output = discriminator(fake_data)
    loss = loss_function(output, ones_tensor(output.size(0)))
    loss.backward()
    optimizer.step()
    return loss


def train_cgan(loss_function, num_epochs=100):
    d_losses = []
    g_losses = []
    it = 0
    for epoch in range(num_epochs):
        print(f'\r epoch: [{epoch+1}/{num_epochs}]', end='')
        for true_batch,_ in dataloader:

            true_data = Variable(true_batch)
            if torch.cuda.is_available():
                true_data = true_data.cuda()

            fake_data = generator(noise(true_data.size(0))).detach()
            discriminator_loss = discriminator_step(d_optim, loss_function, true_data, fake_data)

            fake_data = generator(noise(true_batch.size(0)))
            generator_loss = generator_step(g_optim, loss_function, fake_data)

            print(f'\r epoch: [{epoch+1}/{num_epochs}], '
                  f'discriminator: {float(discriminator_loss.data.cpu())}, '
                  f'generator: {float(generator_loss.data.cpu())}', end='')
            plotter.on_new_point(
                label='train',
                x=it,
                y=np.array([float(discriminator_loss.data.cpu()), float(generator_loss.data.cpu())])
            )
            it += 1
        d_losses.append(float(discriminator_loss.data.cpu()))
        g_losses.append(float(generator_loss.data.cpu()))
        if epoch % log_interval == 0:
            torch.save(discriminator.state_dict(),
                       'saved_models/cifar/discr_' + str(epoch) + '_' + str(num_epochs) + '.pt')
            torch.save(generator.state_dict(),
                       'saved_models/cifar/gen_' + str(epoch) + '_' + str(num_epochs) + '.pt')
            fake_data = fake_data.data.cpu().numpy()
            filename = 'fake_data/cifar/fake_' + str(epoch) + '.csv'
            np.savetxt(filename, fake_data, delimiter=',')
            img_tensor = to_image_tensor(fake_data)
            plotter.on_new_image(label='fake data ', img_tensor=img_tensor, global_step=epoch)
    torch.save(discriminator.state_dict(), 'saved_models/cifar/discr_final.pt')
    torch.save(generator.state_dict(), 'saved_models/cifar/gen_final.pt')
    filename = 'fake_data/cifar/fake_data.csv'
    np.savetxt(filename, fake_data, delimiter=',')
    return d_losses, g_losses


d_losses, g_losses = train_cgan(loss_function, num_epochs)