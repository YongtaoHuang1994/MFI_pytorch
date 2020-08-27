# 这段代码用于推理整组试集, 看模型准确率

from model import LeNet5, LeNet5_Improved
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, transforms
from mfidata import MfiDataset

torch.manual_seed(2020)

# FILE_NAME = "mfi_0.97400.pth"
FILE_NAME = "quantized_1_model.pth"

if __name__ == '__main__':
    batch_size = 16
    test_dataset = MfiDataset(root_dir='./data/test/',
                        names_file='./data/test/test.csv',transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = torch.load("./models/"+FILE_NAME)
    model.eval()

    result = list()

    correct = 0
    _sum = 0

    for idx, (test_x, test_label) in enumerate(test_loader):
        test_x = test_x.unsqueeze(0)
        test_x = test_x.to(torch.float32)    # float32
        test_x = torch.div(test_x, 255)
        predict_y = model(test_x.float()).detach()
        # print(predict_y)
        predict_ys = np.argmax(predict_y, axis=-1)
        # print("predict_y")
        # print(predict_ys.numpy())
        label_np = test_label.numpy()
        # print("label_np")
        # print(label_np)
        _ = predict_ys == test_label
        correct += np.sum(_.numpy(), axis=-1)
        _sum += _.shape[0]
        result.append([predict_ys.numpy(), label_np])

    print('accuracy: {:.5f}'.format(correct / _sum))
    print('\n')
    dt = DataFrame(result)
    dt.to_csv('result.csv', header=None, index=None)
