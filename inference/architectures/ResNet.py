import torch
import torch.nn as nn

class ResNet(nn.Module): # (mejor configuración obtenida)
    def __init__(self, input_size=1, output_size=4, kernel_size=3):
        super().__init__()

        self.conv11 = nn.Conv2d(in_channels=input_size, out_channels=2, kernel_size=kernel_size, padding=1)
        self.bn11 = nn.BatchNorm2d(2) 
        self.conv12 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=kernel_size, padding=1)
        self.bn12 = nn.BatchNorm2d(2) 

        self.ecualizador1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2)
        )

        
        self.conv21 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=kernel_size, padding=1)
        self.bn21 = nn.BatchNorm2d(4)
        self.conv22 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=kernel_size, padding=1)
        self.bn22 = nn.BatchNorm2d(4)

        self.ecualizador2 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=1, stride=1),
            nn.BatchNorm2d(4)
        )


        self.conv31 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=kernel_size, padding=1, stride=1)
        self.bn31 = nn.BatchNorm2d(8)
        self.conv32 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel_size, padding=1, stride=1)
        self.bn32 = nn.BatchNorm2d(8)
        
        self.ecualizador3 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=1, stride=1),
            nn.BatchNorm2d(8)
        )
        

        self.conv41 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernel_size, padding=1, stride=1)
        self.bn41 = nn.BatchNorm2d(16)
        self.conv42 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, padding=1, stride=1)
        self.bn42 = nn.BatchNorm2d(16)

        self.ecualizador4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16)
        )


        self.conv51 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size, padding=1, stride=1)
        self.bn51 = nn.BatchNorm2d(32)
        self.conv52 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=1, stride=1)
        self.bn52 = nn.BatchNorm2d(32)

        self.ecualizador5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32)
        )

        
        

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((3, 4))
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.5) 


        self.fc1 = nn.Linear(in_features=384, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=output_size)

    def forward(self, x):
        layer_input = self.ecualizador1(x)
        x = self.relu(self.bn11(self.conv11(x)))
        x = self.bn12(self.conv12(x))
        x = self.relu(x + layer_input)
        x = self.max_pooling(x)

        layer_input = self.ecualizador2(x)
        x = self.relu(self.bn21(self.conv21(x)))
        x = self.bn22(self.conv22(x))
        x = self.relu(x + layer_input)
        x = self.max_pooling(x)

        layer_input = self.ecualizador3(x)
        x = self.relu(self.bn31(self.conv31(x)))
        x = self.bn32(self.conv32(x))
        x = self.relu(x + layer_input)
        x = self.max_pooling(x)

        layer_input = self.ecualizador4(x)
        x = self.relu(self.bn41(self.conv41(x)))
        x = self.bn42(self.conv42(x))
        x = self.relu(x + layer_input)
        x = self.max_pooling(x)

        layer_input = self.ecualizador5(x)
        x = self.relu(self.bn51(self.conv51(x)))
        x = self.bn52(self.conv52(x))
        x = self.relu(x + layer_input)
        x = self.max_pooling(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x