import torch
import torch.nn as nn
import math

class DenseNet(nn.Module): # version a partir de mi arquitectura vgg + conexiones densas
    def __init__(self, input_size=1, output_size=4, kernel_size=3, k = 4):
        super().__init__()
        self.k = k  # tasa de crecimiento — canales que añade cada conv

        self.conv11 = nn.Conv2d(input_size,       self.k, kernel_size, padding=1)
        self.bn11   = nn.BatchNorm2d(self.k)
        self.conv12 = nn.Conv2d(input_size + self.k,   self.k, kernel_size, padding=1)
        self.bn12   = nn.BatchNorm2d(self.k)
        # tras bloque 1: input_size + 2k canales

        in2 = input_size + 2*self.k
        self.conv21 = nn.Conv2d(in2,       self.k, kernel_size, padding=1)
        self.bn21   = nn.BatchNorm2d(self.k)
        self.conv22 = nn.Conv2d(in2 + self.k,   self.k, kernel_size, padding=1)
        self.bn22   = nn.BatchNorm2d(self.k)
        # tras bloque 2: in2 + 2k canales

        in3 = in2 + 2*self.k
        self.conv31 = nn.Conv2d(in3,           self.k, kernel_size, padding=1)
        self.bn31   = nn.BatchNorm2d(self.k)
        self.conv32 = nn.Conv2d(in3 + self.k,       self.k, kernel_size, padding=1)
        self.bn32   = nn.BatchNorm2d(self.k)
        self.conv33 = nn.Conv2d(in3 + 2*self.k,     self.k, kernel_size, padding=1)
        self.bn33   = nn.BatchNorm2d(self.k)
        # tras bloque 3: in3 + 3k canales

        in4 = in3 + 3*self.k
        self.conv41 = nn.Conv2d(in4,           self.k, kernel_size, padding=1)
        self.bn41   = nn.BatchNorm2d(self.k)
        self.conv42 = nn.Conv2d(in4 + self.k,       self.k, kernel_size, padding=1)
        self.bn42   = nn.BatchNorm2d(self.k)
        self.conv43 = nn.Conv2d(in4 + 2*self.k,     self.k, kernel_size, padding=1)
        self.bn43   = nn.BatchNorm2d(self.k)
        # tras bloque 4: in4 + 3k canales


        in5 = in4 + 3*self.k
        self.conv51 = nn.Conv2d(in5,           self.k, kernel_size, padding=1)
        self.bn51   = nn.BatchNorm2d(self.k)
        self.conv52 = nn.Conv2d(in5 + self.k,       self.k, kernel_size, padding=1)
        self.bn52   = nn.BatchNorm2d(self.k)
        self.conv53 = nn.Conv2d(in5 + 2*self.k,     self.k, kernel_size, padding=1)
        self.bn53   = nn.BatchNorm2d(self.k)
        # tras bloque 5: in5 + 3k canales

        final_channels = in5 + 3*self.k  # 

        self.max_pooling  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 4))
        self.relu    = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(final_channels*3*4, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):

        out = self.relu(self.bn11(self.conv11(x))) #4
        x   = torch.cat([x, out], dim=1)            # 5
        out = self.relu(self.bn12(self.conv12(x))) #4
        x   = torch.cat([x, out], dim=1)  # 9
        x   = self.max_pooling(x) # 9


        out = self.relu(self.bn21(self.conv21(x))) # 4
        x   = torch.cat([x, out], dim=1) #13
        out = self.relu(self.bn22(self.conv22(x))) #4
        x   = torch.cat([x, out], dim=1) #17
        x   = self.max_pooling(x) #17


        out = self.relu(self.bn31(self.conv31(x)))
        x   = torch.cat([x, out], dim=1)
        out = self.relu(self.bn32(self.conv32(x)))
        x   = torch.cat([x, out], dim=1)
        out = self.relu(self.bn33(self.conv33(x)))
        x   = torch.cat([x, out], dim=1)
        x   = self.max_pooling(x)


        out = self.relu(self.bn41(self.conv41(x)))
        x   = torch.cat([x, out], dim=1)
        out = self.relu(self.bn42(self.conv42(x)))
        x   = torch.cat([x, out], dim=1)
        out = self.relu(self.bn43(self.conv43(x)))
        x   = torch.cat([x, out], dim=1)
        x   = self.max_pooling(x)


        out = self.relu(self.bn51(self.conv51(x)))
        x   = torch.cat([x, out], dim=1)
        out = self.relu(self.bn52(self.conv52(x)))
        x   = torch.cat([x, out], dim=1)
        out = self.relu(self.bn53(self.conv53(x)))
        x   = torch.cat([x, out], dim=1)
        x   = self.max_pooling(x)


        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
