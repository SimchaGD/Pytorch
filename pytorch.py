import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

trainSet = torchvision.datasets.FashionMNIST(
    root = "./data/FashionMNIST", #Extract
    train = True,#Extract
    download = True,#Extract
    transform = transforms.Compose([ #Transform
        transforms.ToTensor()
    ])
) 

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Convolutional transformations
        # Zorg dat het aantal out_channels meer en meer wordt
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5)
        
        # Fully Connected layers
        # Zorg dat het aantal out_features minder en minder wordt
        self.fc1 = nn.Linear(in_features = 12*4*4, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60)
        self.out = nn.Linear(in_features = 60, out_features = 10)
        
    def forward(self, t):
        # Implementation of layers
        # (1) input layer
        t = t
        
        # (2) hidden conv layer 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        
        # (3) hidden conv layer 2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)
        
        # (4) hidden linear layer 1
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)
        
        # (5) hidden linear layer 2
        t = self.fc2(t)
        t = F.relu(t)
        
        # (6) output linear layer
        t = self.out(t)
        # Normaal zou je bij de output layer een softmax uitvoeren na een serie van relu operaties
        # De loss/cost functie die we gaan toepassen maakt al impliciet gebruik van de softmax 
        #t = F.softmax(t, dim = 1)             
        return t

network = Network()
sample = next(iter(trainSet))
image, label = sample
pred = network(image.unsqueeze(0))
pred