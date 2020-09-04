# pytorch 支持的第一种剪枝方式


from model import LeNet5
import numpy as np
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from mfidata import MfiDataset
from utils import time_model_evaluation, print_size_of_model

FILE_NAME = "mfi_20_0.99.pth"
PROB_PRUNE = 0.5
PRUNING_FILE_NAME = "pruning_1_model_"+str(PROB_PRUNE)+".pth"
PRUNING_PKL_NAME = "pruning_1_model_"+str(PROB_PRUNE)+".pkl"

if __name__ == '__main__':
    model = torch.load("./models/"+FILE_NAME)
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
    print("==================================")
    print("(1) original model time cost: ")
    time_model_evaluation(model, images)
    print("(1) original model size: ")
    print_size_of_model(model)
    print("==================================")

    print("model dicts before pruning: ")
    print(model.state_dict().keys()) # 序列化修剪的模型

    parameters_to_prune = (
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.fc1, 'weight'),
        (model.fc2, 'weight'),
        (model.fc3, 'weight'),
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=PROB_PRUNE,
    )

    print(
        "Sparsity in conv1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv1.weight == 0))
            / float(model.conv1.weight.nelement())
        )
    )
    print(
        "Sparsity in conv2.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv2.weight == 0))
            / float(model.conv2.weight.nelement())
        )
    )
    print(
        "Sparsity in fc1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc1.weight == 0))
            / float(model.fc1.weight.nelement())
        )
    )
    print(
        "Sparsity in fc2.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc2.weight == 0))
            / float(model.fc2.weight.nelement())
        )
    )
    print(
        "Sparsity in fc3.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc3.weight == 0))
            / float(model.fc3.weight.nelement())
        )
    )
    print(
        "Global sparsity: {:.2f}%".format(
            100. * float(
                torch.sum(model.conv1.weight == 0)
                + torch.sum(model.conv2.weight == 0)
                + torch.sum(model.fc1.weight == 0)
                + torch.sum(model.fc2.weight == 0)
                + torch.sum(model.fc3.weight == 0)
            )
            / float(
                model.conv1.weight.nelement()
                + model.conv2.weight.nelement()
                + model.fc1.weight.nelement()
                + model.fc2.weight.nelement()
                + model.fc3.weight.nelement()
            )
        )
    )

    print("model dicts after pruning: ")
    print(model.state_dict().keys()) # 序列化修剪的模型

    """
    parameters_to_prune = (
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.fc1, 'weight'),
        (model.fc2, 'weight'),
        (model.fc3, 'weight'),
    )
    """
    prune.remove(model.conv1, 'weight')
    prune.remove(model.conv2, 'weight')
    prune.remove(model.fc1, 'weight')
    prune.remove(model.fc2, 'weight')
    prune.remove(model.fc3, 'weight')

    print("after remove sth. ==================================")

    print(model.state_dict().keys()) # 序列化修剪的模型



    torch.save(model, 'models/'+PRUNING_FILE_NAME)
    #torch.save(model.state_dict(), 'models/'+PRUNING_PKL_NAME)


    print("==================================")
    print("(2) pruning model time cost: ")
    time_model_evaluation(model, images)

    print("(2) pruning model modle size: ")
    print_size_of_model(model)
    print("==================================")







