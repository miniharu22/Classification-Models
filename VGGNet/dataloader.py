
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np 

def load_dataset():

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
                                    
    trainset = torchvision.datasets.CIFAR10(root='/data',
                                            train=True,
                                            download=True,
                                            transform=transform)

    testset = torchvision.datasets.CIFAR10(root='/data',
                                        train=False,
                                        download=True,
                                        transform=transform)
                                        
    trainloader = DataLoader(trainset,
                            batch_size=256,
                            shuffle=True,
                            num_workers=2)
    testloader = DataLoader(testset,
                            batch_size=100,
                            shuffle=True,
                            num_workers=2)
    
    return trainloader, testloader