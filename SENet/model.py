import torch.nn as nn

class se_block(nn.Module):  # Squeeze-Excitation Block
    """
    c : Number of I/O channels
    r : Reduction ratio for the bottleneck
    """
    def __init__(self,c,r):
        super(se_block,self).__init__()
        # Squeeze : Global Average Pooling to produce channel descriptors
        self.sq = nn.AdaptiveAvgPool2d(1) 

        # Excitation : Small Bottleneck MLP to capture channel interdependencies
        self.ex = nn.Sequential(          
            nn.Linear(c,int(c/r)),  # channel reduction by r factor
            nn.ReLU(),              # Non-linearity
            nn.Linear(int(c/r),c),  # restore original channel dimension
            nn.Sigmoid()            # scale to 0~1 per channel
        )
    
    def forward(self,x):
        sq = self.sq(x)
        sq = sq.view(sq.size(0),-1)

        ex = self.ex(sq)
        ex = ex.view(ex.size(0), ex.size(1), 1, 1)

        out = x*ex

        return out

        
# Bottleneck residual block with a SE Block
class bottleneck_block(nn.Module):
    """
    i : Number of input channels
    o : Number of base channels before expansion
    s : Stride for the 1st Conv
    e : Expansion factor for the 3rd Conv 
    """
    def __init__(self,i,o,s,e,stage):
        super(bottleneck_block,self).__init__()
        
        # 1x1 Conv to reduce channels
        self.conv1 = nn.Conv2d(i,o,1,s)
        self.bn1 = nn.BatchNorm2d(o)
        self.relu = nn.ReLU()

        # 3x3 Conv for spatial processing
        self.conv2 = nn.Conv2d(o,o,3,1,1)
        self.bn2 = nn.BatchNorm2d(o)

        # 1x1 Conv to expand channels by e factor
        self.conv3 = nn.Conv2d(o,o*e,1,1)
        self.bn3 = nn.BatchNorm2d(o*e)

        # Shortcut Path
        if s == 2 or i==o:
            self.identity = nn.Sequential(
                nn.Conv2d(i,o*e,1,s),
                nn.BatchNorm2d(o*e)
            )
        else :
            self.identity = nn.Sequential()

        # Add SE Block after the main path Conv
        self.se = se_block(o*e,16)

        

    def forward(self,x):
        # Shortcut branch
        identity = self.identity(x) 

        # Main branch : conv1 → BN → ReLU → conv2 → BN → ReLU → conv3 → BN
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)

        # Apply channel weighting
        out = self.se(out)
        
        # Add Skip connection    
        out += identity
        out = self.relu(out)

        return out

# ResNet50 with SE Block
class SE_ResNet50(nn.Module):
    def __init__(self,e=4,num_layers=[3,4,6,3]):
        super(SE_ResNet50,self).__init__()
        
        # Create a sequence of bottleneck blocks for one stage
        def n_blocks(i,o,s,stage):
            layers = []
            layers.append(bottleneck_block(i,o,s,e,stage))

            for _ in range(1,num_layers[stage]):
                layers.append(bottleneck_block(o*e,o,1,e,stage))

            return nn.Sequential(*layers)

        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,7,2,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1)
        )

        # Four stages of bottleneck blocks
        self.stage1 = n_blocks(64,64,1,0)
        self.stage2 = n_blocks(64*e,128,2,1)
        self.stage3 = n_blocks(128*e,256,2,2)
        self.stage4 = n_blocks(256*e,512,2,3)

        self.F = nn.AdaptiveAvgPool2d(1)

        self.FC = nn.Sequential(
            nn.Linear(512*e,10) # output 10 classes
        )


    def forward(self,x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        out = self.F(out)               # shape : (batch, channels, 1, 1)
        out = out.view(out.size(0),-1)  # flatten to (batch, channels)
        out = self.FC(out)              # final logits
        
        return out