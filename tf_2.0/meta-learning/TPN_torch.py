import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Hyperparameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

DATA_PATH = '.\MNISTData'
MODEL_STORE_PATH = '.\pytorch_models\\'

# transforms to apply to the data
trans = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(
    root=DATA_PATH, train=False, transform=trans)

padding = (kernel_size - stride)//2


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(eps=1e-3),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(eps=1e-3),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(eps=1e-3),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.flatten = nn.Flatten()
        # self.drop_out = nn.Dropout()
        # self.fc1 = nn.Linear(7*7*64, 1000)
        # self.fc2 - nn.Linear(1000, 10)

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block2(out)
        encoder_out = self.conv_block3(out)
        # out = self.drop_out(out)
        # out = self.fc1(out)
        # out = self.fc2(out)
        x_temp = x.view(-1, 5, 5, 64)
        net = self.conv_block2(x_temp)
        net = self.conv_block4(net)

        return self.flatten(encoder_out),
