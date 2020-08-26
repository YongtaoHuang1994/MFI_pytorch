# 这段代码用于推理单个样本, 看输出结果

from model import LeNet5, LeNet5_Improved
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import cv2
import torch
from torchvision.datasets import mnist
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, transforms
from mfidata import MfiDataset

torch.manual_seed(2020)

if __name__ == '__main__':
    batch_size = 1
    test_dataset = MfiDataset(root_dir='./data/test/',
                        names_file='./data/test/test.csv',transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = torch.load("./models/mfi_0.81200.pth") # 98.74
    
    correct = 0
    _sum = 0

    model.eval()

    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images = images.unsqueeze(0)

    images = images.to(torch.float32)    # float32
    images = torch.div(images, 255)

    print("images: ")
    print(images)
    print(images.size())
    print("labels: ")
    print(labels)

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(images)
        print("output: ")
        print(output)
        #print(output.size())
        #print(type(output))
        max_value, max_idx = torch.max(output,dim=1)
        #print(max_value)
        print("predict label")
        print(max_idx)


