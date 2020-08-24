# pytorch 支持的第二种量化方式 static quantization 
# TODO 还没实现完全

import os
import time
import torch
import torch.quantization
import torch.nn as nn
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.quantization import QuantStub, DeQuantStub
from model import LeNet5


# # Setup warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

# Specify random seed for repeatable results
torch.manual_seed(191009)

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def time_model_evaluation(model, test_data):
    s = time.time()
    output = model.forward(test_data)
    elapsed = time.time() - s
    print('elapsed time (seconds): ', elapsed)

if __name__ == '__main__':

    # prepare data
    batch_size = 256
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    float_model = torch.load("./models/mnist_0.98710.pth")
