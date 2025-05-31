import torch.nn as nn

class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5,self).__init__()

        # <<< 이 한 줄이 반드시 "3 → 6" 이어야 합니다! >>>
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1),  # in_channels=3 (RGB)
            nn.ReLU()
        )
        self.s2 = nn.AvgPool2d(2, 2)
        self.c3 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1),
            nn.ReLU()
        )
        self.s4 = nn.AvgPool2d(2, 2)

        self.c5 = nn.Sequential(
            nn.Conv2d(16, 120, 5, 1),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(84, 10)
        )

    def forward(self, x):
        c1 = self.c1(x)      # → [B, 6, 28, 28]
        s2 = self.s2(c1)     # → [B, 6, 14, 14]
        c3 = self.c3(s2)     # → [B, 16, 10, 10]
        s4 = self.s4(c3)     # → [B, 16, 5, 5]
        c5 = self.c5(s4)     # → [B, 120, 1, 1]
        c5 = c5.view(c5.size(0), -1)  # → [B, 120]
        fc1 = self.fc1(c5)   # → [B, 84]
        fc2 = self.fc2(fc1)  # → [B, 10]
        return fc2
