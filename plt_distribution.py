# plot weight distribution
import torch
import numpy as np
import matplotlib.pyplot as plt

MODLE_LOCATION = "./models/mfi_0.97400.pth"
TENSOR_NAME = "conv1.weight"

if __name__ == '__main__':
    model = torch.load(MODLE_LOCATION)
    params=model.state_dict() 
    tensor_value = params[TENSOR_NAME]
    tensor_value_np = tensor_value.numpy()
    print(type(tensor_value_np))
    tensor_value_np = tensor_value_np.flatten()
    bins = np.arange(-1, 1, 0.1)
    plt.hist(tensor_value_np,bins) 
    plt.title("histogram") 
    plt.show()