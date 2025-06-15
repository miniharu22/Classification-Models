import torch.nn as nn

# Swish Activation class
class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Swich Function : x * sigmoid(x)
        out = self.sigmoid(x)
        return x * out

# Squeeze-Excitation Block
class se_block(nn.Module):
    def __init__(self,c,r=4):
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


# Idea from MobileNetv2
class MBConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,e):
        super(MBConv,self).__init__()

        # 1x1 expansion conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels*e,1,1),
            nn.BatchNorm2d(in_channels*e),
            swish()
        )

        # Depthwise conv
        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels*e,in_channels*e,kernel_size,stride,int((kernel_size-1)/2),groups=in_channels*e),
            nn.BatchNorm2d(in_channels*e),
            swish()
        )

        # SE Block + Swish Activation
        self.se_block = nn.Sequential(
            se_block(in_channels*e),
            swish()
        )

        # 1x1 projection conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels*e,out_channels,1,1),
            nn.BatchNorm2d(out_channels)
        )

        # residual connection only if shape is preserved
        self.residual = stride==1 and in_channels == out_channels
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.dconv(out)
        out = self.se_block(out)
        out = self.conv2(out)

        # Skip connection
        if self.residual:
            out += x

        return out

# EfficientNet-B0 Architecture (w/o Scale-up)
class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,3,2,1),
            nn.BatchNorm2d(32),
            swish()
        )
        
        # Series of MBConv with increasing depth & width
        self.mbconv1 = MBConv(32,16,3,1,1)

        self.mbconv2 = nn.Sequential(
            MBConv(16,24,3,1,6),
            MBConv(24,24,3,2,6)
        )

        self.mbconv3 = nn.Sequential(
            MBConv(24,40,5,1,6),
            MBConv(40,40,5,2,6)
        )

        self.mbconv4 = nn.Sequential(
            MBConv(40,80,3,1,6),
            MBConv(80,80,3,1,6),
            MBConv(80,80,3,2,6)
        )

        self.mbconv5 = nn.Sequential(
            MBConv(80,112,5,1,6),
            MBConv(112,112,5,1,6),
            MBConv(112,112,5,1,6)
        )

        self.mbconv6 = nn.Sequential(
            MBConv(112,192,5,1,6),
            MBConv(192,192,5,1,6),
            MBConv(192,192,5,1,6),
            MBConv(192,192,5,2,6)
        )

        self.mbconv7 = MBConv(192,320,3,1,6)

        self.conv2 = nn.Sequential(
            nn.Conv2d(320,1280,1,1),
            nn.BatchNorm2d(1280),
            swish(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.FC = nn.Sequential(
            nn.Linear(1280,10)
        )


    def forward(self,x):
        out = self.conv1(x)
        out = self.mbconv1(out)
        out = self.mbconv2(out)
        out = self.mbconv3(out)
        out = self.mbconv4(out)
        out = self.mbconv5(out)
        out = self.mbconv6(out)
        out = self.mbconv7(out)
        out = self.conv2(out)
        out = out.view(out.size(0),-1)
        out = self.FC(out)

        return out
