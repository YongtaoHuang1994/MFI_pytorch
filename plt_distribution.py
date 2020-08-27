# plot weight distribution
import torch
import numpy as np
import matplotlib.pyplot as plt

MODLE_LOCATION = "./models/mfi_0.97400.pth"
MODLE_LOCATION_QUAN = "./models/quantized_1_model.pth"
TENSOR_NAME = "fc1.weight"

def plot_distribution(model_name, tensor_set, range_left, range_right, resolution):
    model = torch.load(model_name)
    print(model)
    params = model.state_dict()
    tensor_value = params[TENSOR_NAME]
    tensor_value_np = tensor_value.numpy()
    tensor_value_np = tensor_value_np.flatten()
    bins = np.arange(range_left, range_right, resolution)
    plt.hist(tensor_value_np,bins) 
    plt.title("histogram") 
    plt.show()


if __name__ == '__main__':
    plot_distribution(MODLE_LOCATION, TENSOR_NAME, -1, 1, 0.01)



    # 用来打印单个节点的量化前后权值分布
    '''
    model = torch.load(MODLE_LOCATION)
    print(model)
    params = model.state_dict()
    tensor_value = params[TENSOR_NAME]
    weight_prepack, col_offsets, scale, zero_point = torch.fbgemm_linear_quantize_weight(tensor_value)
    print(weight_prepack)
    print(col_offsets)
    print(scale)
    print(zero_point)
    print(weight_prepack.size())
    tensor_value_np = weight_prepack.numpy()
    tensor_value_np = tensor_value_np.flatten()
    bins = np.arange(-255, 255, 1)
    plt.hist(tensor_value_np,bins) 
    plt.title("histogram") 
    plt.show()
    '''
    

    
