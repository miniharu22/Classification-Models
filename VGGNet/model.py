import torch.nn as nn
from torch.nn.modules.dropout import Dropout

# VGG model definition
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        # Initialize the VGG model with the specified configuration
        self.features = self._make_layers(cfg[vgg_name])

        self.fc_layer = nn.Sequential(
            # Cifar-10 Size = 32x32  
            nn.Linear(512*1*1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(1000, num_classes)
            )
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)

        return out   
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), 
                        nn.BatchNorm2d(x),   
                        nn.ReLU(inplace=True)]
                in_channels = x

        return nn.Sequential(*layers)