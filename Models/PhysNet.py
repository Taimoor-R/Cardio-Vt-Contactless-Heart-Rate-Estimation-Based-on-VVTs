import torch
import torch.nn as nn

class PhysNet(nn.Module):
    def __init__(self, frames=128):
        super(PhysNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.conv5 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.maxpool_spa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.maxpool_spa_tem= nn.MaxPool3d((2, 2, 2), stride=2)
        self.pool_spa = nn.AdaptiveAvgPool3d((frames, 1, 1))
        
    def forward(self, x):  # Batch_size*[3, T, 128,128]
        [batch, channel, length, width, height] = x.shape

        x = self.conv1(x) 
        x = self.maxpool_spa(x) 
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool_spa_tem(x)
        for i in range(2):
            x = self.conv4(x)

        x = self.maxpool_spa_tem(x)

        for i in range(2):
            x = self.conv4(x)
        x = self.maxpool_spa(x)
        
        for i in range(2):
            x = self.conv4(x)
        
        for i in range(2):
            x = self.upsample(x) 

        x = self.pool_spa(x)
        x = self.conv5(x) 

        rPPG = x.view(-1, length)

        return rPPG
