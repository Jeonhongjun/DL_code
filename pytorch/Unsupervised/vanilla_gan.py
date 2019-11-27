import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torch.utils.data
from torch.autograd import Variable

batch_size = 128

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../Unsupervised/data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../Unsupervised/data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

device = torch.device("cuda")

class generator(nn.Module):

    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 784)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.tan = nn.Tanh()

    def forward(self, x):
        x = x.view(x.size(0), 100)
        x = self.fc1(x)
        x = self.fc2(self.relu(x))
        x = self.fc3(self.relu(x))
        x = self.fc4(self.relu(x))

        return self.tan(x)

class discriminator(nn.Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.sigmoid(x)

        return x.view(x.size(0), -1)

generator = generator().to(device)
discriminator = discriminator().to(device)

criterion = nn.BCELoss()
lr = 0.0002
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

def train_discriminator(discriminator, images, real_labels, fake_images, fake_labels):

    d_optimizer.zero_grad()
    real_loss = criterion(discriminator(images), real_labels)
    real_score = discriminator(images)

    outputs = discriminator(fake_images)
    fake_loss = criterion(outputs, fake_labels)

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()

    return d_loss, real_score, fake_loss

def train_generator(generator, discriminator_outputs, real_labels):

    g_optimizer.zero_grad()
    g_loss = criterion(discriminator_outputs, real_labels)
    d_loss.backward()
    g_optimizer.step()

    return g_loss

# create figure for plotting
size_figure_grid = int(math.sqrt(num_test_samples))
fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6, 6))
for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
    ax[i,j].get_xaxis().set_visible(False)
    ax[i,j].get_yaxis().set_visible(False)

# set number of epochs and initialize figure counter
num_epochs = 200
num_batches = len(train_loader)
num_fig = 0

for epoch in range(num_epochs):
    for n, (images, _) in enumerate(train_loader):
        images = Variable(images.cuda())
        real_labels = Variable(torch.ones(images.size(0)).cuda())

        # Sample from generator
        noise = Variable(torch.randn(images.size(0), 100).cuda())
        fake_images = generator(noise)
        fake_labels = Variable(torch.zeros(images.size(0)).cuda())

        # Train the discriminator
        d_loss, real_score, fake_score = train_discriminator(discriminator, images, real_labels, fake_images, fake_labels)

        # Sample again from the generator and get output from discriminator
        noise = Variable(torch.randn(images.size(0), 100).cuda())
        fake_images = generator(noise)
        outputs = discriminator(fake_images)

        # Train the generator
        g_loss = train_generator(generator, outputs, real_labels)

        if (n+1) % 100 == 0:
            test_images = generator(test_noise)

            for k in range(num_test_samples):
                i = k//4
                j = k%4
                ax[i,j].cla()
                ax[i,j].imshow(test_images[k,:].data.cpu().numpy().reshape(28, 28), cmap='Greys')
            display.clear_output(wait=True)
            display.display(plt.gcf())

            plt.savefig('results/mnist-gan-%03d.png'%num_fig)
            num_fig += 1
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, '
                  'D(x): %.2f, D(G(z)): %.2f'
                  %(epoch + 1, num_epochs, n+1, num_batches, d_loss.data[0], g_loss.data[0],
                    real_score.data.mean(), fake_score.data.mean()))

fig.close()
