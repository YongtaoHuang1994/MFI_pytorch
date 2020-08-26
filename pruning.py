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
from utils import time_model_evaluation

FILE_NAME = "mfi_0.96400.pth"
PRUNING_FILE_NAME = "pruning_1_model.pth"

if __name__ == '__main__':
    model = torch.load("./models/"+FILE_NAME)
    module = model.conv1
    # print(module)
    # print(list(module.named_parameters()))
    # print(list(module.named_buffers()))
    # 随机裁剪weight
    # prune.random_unstructured(module, name="weight", amount=0.3)
    # print(list(module.named_parameters()))
    # print(list(module.named_buffers()))
    # print(module.weight)
    # 返回裁剪了哪些Tensor
    # print("已经被裁剪的Tenosr")
    # print(module._forward_pre_hooks)
    # 裁剪bias
    # prune.l1_unstructured(module, name="bias", amount=3)
    # print(list(module.named_parameters()))
    # print(module._forward_pre_hooks)
    # print(module.bias)
    # L2裁剪weight
    # prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)
    print("model dicts before pruning: ")
    print(model.state_dict().keys())

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
        amount=0.3,
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
    print(model.state_dict().keys())

    torch.save(model, 'models/'+PRUNING_FILE_NAME)

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
    print("pruning time cost: ")
    time_model_evaluation(model, images)







