# 训练脚本
# 20分类 torch.manual_seed(2020) batch_size = 128 epoch = 200 能训练出0.99

from model import LeNet5, LeNet5_Improved, LeNet5_Improved_V2, LeNet5_Improved_V3
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import cv2
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, transforms
from mfidata import MfiDataset
from torchvision.datasets import mnist
from tensorboardX import SummaryWriter

torch.manual_seed(2020)
TYPE_NUM = 20

if __name__ == '__main__':
    batch_size = 128

    '''
    train_dataset = mnist.MNIST(root='./mnist/train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./mnist/test', train=False, transform=ToTensor())
    '''

    train_dataset = MfiDataset(root_dir='./data/train/',
                        names_file='./data/train/train.csv',transform=ToTensor())
    test_dataset = MfiDataset(root_dir='./data/test/',
                        names_file='./data/test/test.csv',transform=ToTensor())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = LeNet5_Improved_V2()

    # optim = SGD(model.parameters(), lr=1e-1) # sgd
    # optim = SGD(model.parameters(), lr=1e-1, momentum=0.9) # sgd_momentum 一般来说sgd_momentum都比sgd强
    # optim = RMSprop(model.parameters(), lr=LR, alpha=0.9) # rmsprop 叶博没怎么用过 不需要动态修改学习率
    optim = Adam(model.parameters(), lr=0.001, betas=(0.9,0.999), eps=1e-08, weight_decay=0) # adam 不需要动态修改学习率
    
    cross_error = CrossEntropyLoss()
    epoch = 200

    w_graph = SummaryWriter('tfboard/graph')
    w_acc = SummaryWriter('tfboard/acc')
    input = torch.rand(1, 1, 28, 28)
    w_graph.add_graph(model,(input,))

    for _epoch in range(epoch):
        print("This is epoch ", _epoch)
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            label_np = np.zeros((train_label.shape[0], 10))
            optim.zero_grad()
            predict_y = model(train_x.float())
            _error = cross_error(predict_y, train_label.long())
            if idx % 100 == 0:
                print('idx: {}, _error: {}'.format(idx, _error))
            _error.backward()
            optim.step()

        correct = 0
        _sum = 0

        model.eval()
        for idx, (test_x, test_label) in enumerate(test_loader):
        
            predict_y = model(test_x.float()).detach()
            predict_ys = np.argmax(predict_y, axis=-1)
            #label_np = test_label.numpy()
            _ = predict_ys == test_label
            correct += np.sum(_.numpy(), axis=-1)
            _sum += _.shape[0]

        w_acc.add_scalar('Train/Acc', correct/_sum, _epoch)
        print('accuracy: {:.5f}'.format(correct / _sum))
        torch.save(model, 'models/mfi_20_{:.5f}.pth'.format(correct / _sum))
        
