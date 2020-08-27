# LeNet5经典模型和用了dropout和BN的改进版本

import torch
import torch.nn
import torch.nn.functional as F

class ConvBNReLU(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=2, bias=True):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias),
            torch.nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            torch.nn.ReLU(inplace=False)
        )


class LeNet5_Improved_V3(torch.nn.Module):
     
    def __init__(self):   
        super(LeNet5_Improved_V3, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.bn1 = torch.nn.BatchNorm2d(6)
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(16*5*5, 180)
        self.dropout1 = torch.nn.Dropout(0.2) # 0.25是丢掉 p=0 保留所有神经元; p=1  
        self.fc2 = torch.nn.Linear(180, 100)  # (120, 84)
        self.dropout2 = torch.nn.Dropout(0.2) # 
        self.fc3 = torch.nn.Linear(100, 20) # mnist有10个输出(84, 10)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.max_pool_2(x)
        x = x.view(-1, 16*5*5)
        x = torch.nn.functional.relu(self.dropout1(self.fc1(x)))
        x = torch.nn.functional.relu(self.dropout2(self.fc2(x)))
        x = self.fc3(x)
        
        return x

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)


class LeNet5_Improved_V2(torch.nn.Module):
     
    def __init__(self):   
        super(LeNet5_Improved_V2, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(16*5*5, 180)
        self.dropout1 = torch.nn.Dropout(0.2) # 0.25是丢掉 p=0 保留所有神经元; p=1  
        self.fc2 = torch.nn.Linear(180, 100)  # (120, 84)
        self.dropout2 = torch.nn.Dropout(0.2) # 
        self.fc3 = torch.nn.Linear(100, 20) # mnist有10个输出(84, 10)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(-1, 16*5*5)
        x = torch.nn.functional.relu(self.dropout1(self.fc1(x)))
        x = torch.nn.functional.relu(self.dropout2(self.fc2(x)))
        x = self.fc3(x)
        
        return x



class LeNet5_Improved(torch.nn.Module):
     
    def __init__(self):   
        super(LeNet5_Improved, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.bn1 = torch.nn.BatchNorm2d(6)
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(16*5*5, 180)
        self.dropout1 = torch.nn.Dropout(0.2) # 0.25是丢掉 p=0 保留所有神经元; p=1  
        self.fc2 = torch.nn.Linear(180, 100)  # (120, 84)
        self.dropout2 = torch.nn.Dropout(0.2) # 
        self.fc3 = torch.nn.Linear(100, 20) # mnist有10个输出(84, 10)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.max_pool_2(x)
        x = x.view(-1, 16*5*5)
        x = torch.nn.functional.relu(self.dropout1(self.fc1(x)))
        x = torch.nn.functional.relu(self.dropout2(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class LeNet5(torch.nn.Module):
     
    def __init__(self):   
        super(LeNet5, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(16*5*5, 120)
        self.fc2 = torch.nn.Linear(120, 84) 
        self.fc3 = torch.nn.Linear(84, 10)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(-1, 16*5*5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x



