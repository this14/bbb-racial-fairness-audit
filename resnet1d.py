import torch
import torch.nn as nn

class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNet1D34(nn.Module):
    # modified 1D-ResNet-34: [3,4,6,3] blocks
    def __init__(self, in_channels=12, num_classes=1, base_width=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_width, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        widths = [base_width, base_width*2, base_width*4, base_width*8]
        blocks = [3,4,6,3]
        layers = []
        in_ch = base_width
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n):
                stride = 2 if (j==0 and i>0) else 1
                layers.append(BasicBlock1D(in_ch, w, stride=stride))
                in_ch = w
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.permute(0,2,1)
        x = self.stem(x)
        x = self.layers(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)
