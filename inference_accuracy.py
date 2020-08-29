# 这段代码用于推理整组试集, 看模型准确率

from model import LeNet5, LeNet5_Improved
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, transforms
from mfidata import MfiDataset

torch.manual_seed(2020)

FILE_NAME = "mfi_0.50600.pth"
# FILE_NAME = "quantized_1_model.pth"
# FILE_NAME = "pruning_1_model.pth"

def infer_accuracy(model, test_loader):
    model.eval()
    result = list()
    correct = 0
    _sum = 0

    for idx, (test_x, test_label) in enumerate(test_loader):
        test_x = test_x.unsqueeze(1)
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

def infer_one_sampele(model, images, labels):
    model.eval()
    print("labels: ")
    print(labels)
    predict_y = model(images.float()).detach()
    print(predict_y)
    predict_ys = np.argmax(predict_y, axis=-1)
    print("predicted label: ")
    print(predict_ys.numpy())

def infer_one_sampele_V2(model, images, labels):
    model.eval()
    print("labels: ")
    print(labels)
    with torch.no_grad():
        output = model.forward(images.float())
        print("output: ")
        print(output)
        max_value, max_idx = torch.max(output, dim=1)
        print("predicted label")
        print(max_idx)


if __name__ == '__main__':
    batch_size = 1
    test_dataset = MfiDataset(root_dir='./data/test/',
                        names_file='./data/test/test.csv', transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = torch.load("./models/"+FILE_NAME)

    #infer_accuracy(model, test_loader)
    
    dataiter = iter(test_loader)
    image, label = dataiter.next() # (1,28,28)
    # TODO 数据可视化函数
    print(image.size())
    plt.imshow(image.squeeze(0), cmap='gray') # squeeze(i)表示去除第i个维度(28,28)
    plt.show()

    image = image.unsqueeze(0) # unsqueeze(i)表示在i添加一个维度 (1,1,28,28)
    print(image.size())
    

    image = image.to(torch.float32)    # float32
    image = torch.div(image, 255)

    print("1=============================")

    infer_one_sampele(model, image, label)

    print("2=============================")

    infer_one_sampele_V2(model, image, label)
    
    

