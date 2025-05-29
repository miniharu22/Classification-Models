import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np 

def load_dataset():
    train_set = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transforms.ToTensor())
    test_set = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transforms.ToTensor())

    batch_size = 128

    train_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in train_set]
    train_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in train_set]

    train_meanR = np.mean([m[0] for m in train_meanRGB])
    train_meanG = np.mean([m[1] for m in train_meanRGB])
    train_meanB = np.mean([m[2] for m in train_meanRGB])
    train_stdR = np.mean([s[0] for s in train_stdRGB])
    train_stdG = np.mean([s[1] for s in train_stdRGB])
    train_stdB = np.mean([s[2] for s in train_stdRGB])

    train_transformer = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[train_meanR, train_meanG, train_meanB], 
                                                             std=[train_stdR, train_stdG, train_stdB])])

    train_set.transform = train_transformer
    test_set.transform = train_transformer

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader, testloader