import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, dropout = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.conv2 = nn.Conv2d(output_dim, output_dim, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(output_dim)
        
        self.downsample = nn.Conv2d(input_dim, output_dim, 1, stride, 0) if input_dim != output_dim or stride!= 1 else None
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x + residual
        return x
    
    
class Resnet(nn.Module):
    def __init__(self, classes_num=90) -> None:
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(), 
            nn.MaxPool2d(3, 2, 1)
        )
        self.block_group1 = nn.Sequential(
            ResidualBlock(64, 64, 1, 0.1),
            ResidualBlock(64, 64, 1, 0.1),
            ResidualBlock(64, 64, 1, 0.1),
        )
        self.block_group2 = nn.Sequential(
            ResidualBlock(64, 128, 2, 0.1),
            ResidualBlock(128, 128, 1, 0.1),
            ResidualBlock(128, 128, 1, 0.1),
            ResidualBlock(128, 128, 1, 0.1),
        )
        self.block_group3 = nn.Sequential(
            ResidualBlock(128, 256, 2, 0.1),
            ResidualBlock(256, 256, 1, 0.1),
            ResidualBlock(256, 256, 1, 0.1),
            ResidualBlock(256, 256, 1, 0.1),
            ResidualBlock(256, 256, 1, 0.1),
            ResidualBlock(256, 256, 1, 0.1),
        )
        self.block_group4 = nn.Sequential(
            ResidualBlock(256, 512, 2, 0.1),
            ResidualBlock(512, 512, 1, 0.1),
            ResidualBlock(512, 512, 1, 0.1),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, classes_num)
        
    def forward(self, x):
        x = self.in_conv(x)
        x = self.block_group1(x)
        x = self.block_group2(x)
        x = self.block_group3(x)
        x = self.block_group4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.fc(x)
        
    
        
        
        