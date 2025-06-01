import torch.nn as nn

class BottleNeck(nn.Module):
    
    # ResNet Bottleneck block
    # Expansion factor for the bottleneck
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        # Shortcut connection
        self.shortcut = nn.Sequential()

        # If the input and output dimensions are not the same, we need to adjust the shortcut connection
        # this is done by a 1x1 convolution
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
    # end of __init__
    # Forward pass through the block
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_classes=10):
        super().__init__()

        # Initialize the ResNet model with the specified block type and number of classes
        # The initial number of input channels is set to 64
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    
        self.conv2_x = self._make_layer(block, 64, 3, 1)
        self.conv3_x = self._make_layer(block, 128, 4, 2)
        self.conv4_x = self._make_layer(block, 256, 6, 2)
        self.conv5_x = self._make_layer(block, 512, 3, 2)
        # Adaptive average pooling to reduce the feature map to 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layer to output the final class scores
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    # Create a layer of blocks
    # Each layer consists of multiple blocks, where the first block may have a different stride
    # to downsample the feature maps, while subsequent blocks maintain the same spatial dimensions
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x