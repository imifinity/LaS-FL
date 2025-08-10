import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models


class MLP(nn.Module):
    ''' A multilayer perceptron for n-class classification. '''
    def __init__(self, input_size=3*32*32, hidden_size=256, num_classes=10):
        super(MLP, self).__init__() 
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        ''' 
        Forward pass.
        Inputs: (batch_size, 3, 32, 32)
        Outputs: (batch_size, 10)
        '''
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class CNN(nn.Module):
    ''' A convolutional neural network for 10-class classification (CIFAR-10). '''
    def __init__(self, n_channels=3, n_classes=10, input_size=32):
        super(CNN, self).__init__() 
        self.conv1 = nn.Conv2d(n_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        dim = self.compute_output(input_size)

        self.fc1 = nn.Linear(16 * dim * dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def compute_output(self, input):
        x = input - 5 + 1       # Conv1
        x = x // 2              # Pool1
        x = x - 5 + 1           # Conv2
        x = x // 2              # Pool2
        return x

    def forward(self, x):
        ''' 
        Forward pass.
        Inputs: (batch_size, 3, 32, 32)
        Outputs: (batch_size, 10)
        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResNet18(nn.Module):
    ''' ResNet-18 for 200-class classification (TinyImageNet). '''
    def __init__(self, n_channels=3, n_classes=200, input_size=64):
        super(ResNet18TinyImageNet, self).__init__()
        self.model = models.resnet18(weights=None) # Load ResNet-18 without pretrained weights
        self.model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False) # Modify for 64x64 images
        self.model.maxpool = nn.Identity()  # Remove pooling
        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes) # Adjust final layer for 200 classes

    def forward(self, x):
        return self.model(x)