# pytorch 支持的第二种量化方式 static quantization 
# 参考 https://pytorch.apachecn.org/docs/1.4/45.html
# 在 PyTorch 中使用Eager模式进行静态量化
# TODO 插入features 使用集成化的ConvBNReLU
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
from mfidata import MfiDataset


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


FILE_NAME = "mfi_0.97400.pth"
QUANTIZATION_FILE_NAME = "quantized_2_model.pth"

def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            image = image.unsqueeze(1)
            image = image.to(torch.float32)    # float32
            image = torch.div(image, 255)
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            if cnt >= neval_batches:
                return cnt/neval_batches
    return cnt/neval_batches

if __name__ == '__main__':

    # prepare data
    batch_size = 16

    train_dataset = MfiDataset(root_dir='./data/train/',
                        names_file='./data/train/train.csv',transform=ToTensor())
    test_dataset = MfiDataset(root_dir='./data/test/',
                        names_file='./data/test/test.csv',transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    num_calibration_batches = 50

    model = torch.load("./models/"+FILE_NAME).to('cpu')
    model.eval()

    # Fuse Conv, bn and relu
    model.fuse_model()

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    model.qconfig = torch.quantization.default_qconfig
    print(model.qconfig)
    torch.quantization.prepare(model, inplace=True)

    # Calibrate first
    #print('Post Training Quantization Prepare: Inserting Observers')
    #print('\n Inverted Residual Block:After observer insertion \n\n', model.features[1].conv)

    criterion = nn.CrossEntropyLoss()
    # Calibrate with the training set
    evaluate(model, criterion, train_loader, neval_batches=num_calibration_batches)
    print('Post Training Quantization: Calibration done')

    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    #print('Post Training Quantization: Convert done')
    #print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',model.features[1].conv)

    print("Size of model after quantization")
    print_size_of_model(model)

    torch.save(model, 'models/'+QUANTIZATION_FILE_NAME)


