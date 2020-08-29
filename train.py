# 训练脚本
# TODO


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
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(2020)

TYPE_NUM = 20

if __name__ == '__main__':
    batch_size = 1
    train_dataset = MfiDataset(root_dir='./data/train/',
                        names_file='./data/train/train.csv',transform=ToTensor())
    test_dataset = MfiDataset(root_dir='./data/test/',
                        names_file='./data/test/test.csv',transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    #model = LeNet5()
    model = LeNet5_Improved()

    w_graph = SummaryWriter('tfboard/graph')
    w_acc = SummaryWriter('tfboard/acc')
    input = torch.rand(1, 1, 28, 28)
    #with SummaryWriter() as w:
    w_graph.add_graph(model,(input,))
    
    # optim = SGD(model.parameters(), lr=1e-1) # sgd
    # optim = SGD(model.parameters(), lr=1e-1, momentum=0.9) # sgd_momentum 一般来说sgd_momentum都比sgd强
    # optim = RMSprop(model.parameters(), lr=LR, alpha=0.9) # rmsprop 叶博没怎么用过 不需要动态修改学习率
    optim = Adam(model.parameters(), lr=0.001, betas=(0.9,0.999), eps=1e-08, weight_decay=0) # adam 不需要动态修改学习率
    
    cross_error = CrossEntropyLoss()
    epoch = 500

    acc = np.zeros(epoch)

    for _epoch in range(epoch):

        # training   
        model.train() # 模型设置为训练模式，动态修改学习参数     
        for idx, (train_x, train_label) in enumerate(train_loader):
            
            train_x = train_x.unsqueeze(1)
            train_x = train_x.to(torch.float32)    # float32
            train_x = torch.div(train_x, 255)
            label_np = np.zeros((1, TYPE_NUM)) 
            optim.zero_grad() # 清空梯度避免影响下一轮训练
            predict_y = model(train_x.float())
            _error = cross_error(predict_y, train_label.long())
            _error.backward() # 误差反向传播
            optim.step() # 梯度赋值

        correct = 0
        _sum = 0
        
        # inference
        model.eval()# 模型设置为预测模式，锁定学习参数
        for idx, (test_x, test_label) in enumerate(test_loader):
            
            test_x = test_x.unsqueeze(1)
            test_x = test_x.to(torch.float32)    # float32
            test_x = torch.div(test_x, 255)
            predict_y = model(test_x.float()).detach()
            predict_ys = np.argmax(predict_y, axis=-1)
            label_np = test_label.numpy()
            _ = predict_ys == test_label
            correct += np.sum(_.numpy(), axis=-1)
            _sum += _.shape[0]

        print('This is epochs: {}'.format(_epoch))
        print('accuracy: {:.5f}'.format(correct / _sum))
        print('\n')

        torch.save(model, 'models/mfi_{:.5f}.pth'.format(correct / _sum))
        acc[_epoch] = correct/_sum
        
        print(type(correct/_sum))
        print(type(_epoch))
        xx = correct/_sum
        yy = _epoch
        w_acc.add_scalar('Train/Acc', xx, yy)

    print(acc)
    dt = DataFrame(acc)
    dt.to_csv('./models/Result.csv')
    
    
    


