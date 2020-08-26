from model import LeNet5_Improved
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import cv2
import torch
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import Dataset, DataLoader
from mfidata import MfiDataset

torch.manual_seed(2020)

if __name__ == '__main__':
    batch_size = 1
    train_dataset = MfiDataset(root_dir='./data/train/',
                        names_file='./data/train/train.csv',transform=ToTensor())
    test_dataset = MfiDataset(root_dir='./data/test/',
                        names_file='./data/test/test.csv',transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    '''
    for x in enumerate(train_loader):
        print(x)
    '''

    traindataiter = iter(train_loader)
    x = traindataiter.next()
    print(x[1])
    '''
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(x)
    print(type(x))
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(x[0])
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(x[1])
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(type(x[0]))
    print(x[0].size())
    yy = x[0].unsqueeze(0)
    print(yy.size())
    '''

    '''
    for idx, (train_x, train_label) in enumerate(train_loader):
        train_x = train_x.unsqueeze(0)
        label_np = np.zeros((train_label.shape[0], 10))
        print(train_label)
        #print(train_label.shape[0])
    '''

    for idx, (test_x, test_label) in enumerate(test_loader):
        test_x = test_x.unsqueeze(0)
        label_np = np.zeros((test_label.shape[0], 10))
        print(test_label)