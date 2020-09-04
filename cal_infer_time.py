# pytorch 用于计算推理时间

import os
import time
import torch
import cv2
import torch.quantization
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from mfidata import MfiDataset
from utils import time_model_evaluation, print_size_of_model

# FILE_NAME = "mfi_20_0.99.pth"
FILE_NAME = "quantized_1_model.pth"

if __name__ == '__main__':
    model = torch.load("./models/"+FILE_NAME)
    #model.to('cpu')
 
    torch.set_num_threads(1)
    batch_size = 1
    test_dataset = MfiDataset(root_dir='./data/test/',
                        names_file='./data/test/test.csv',transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images = images.unsqueeze(0)
    images = images.to(torch.float32)    # float32
    images = torch.div(images, 255)

    print(FILE_NAME+" model time cost: ")
    time_model_evaluation(model, images)
