# pytorch 支持的第一种量化方式 dynamic quantization

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


FILE_NAME = "mfi_0.96400.pth"
QUANTIZATION_FILE_NAME = "quantized_1_model.pth"



if __name__ == '__main__':
    model = torch.load("./models/"+FILE_NAME)
    print("info of model: ")
    print(model)
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Conv2d,nn.Linear}, dtype=torch.qint8
    )
    print("info of quantized_model: ")
    print(quantized_model)

    # TODO calc size
    print("original model size: ")
    print_size_of_model(model)
    print("quantizedmodel modle size: ")
    print_size_of_model(quantized_model)
    torch.save(quantized_model, 'models/'+QUANTIZATION_FILE_NAME)

    # TODO calc time
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

    print("original model time cost: ")
    time_model_evaluation(model, images)
    print("quantizedmodel time cost: ")
    time_model_evaluation(quantized_model, images)





