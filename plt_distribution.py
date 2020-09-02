# plot weight distribution
import torch
import numpy as np
import matplotlib.pyplot as plt

MODLE_LOCATION = "./models_good/mfi_20_0.99.pth"
MODLE_LOCATION_QUAN = "./models_good/quantized_1_model.pth"
TENSOR_NAME = ["fc1.weight","fc2.weight","fc3.weight","conv1.weight","conv2.weight"]

def plot_distribution(model_name, tensor_set, range_left, range_right, resolution):
    model = torch.load(model_name)
    print(model)
    params = model.state_dict()
    tensor_value_np_all = np.zeros(1)
    for _tensor_name in TENSOR_NAME:
        tensor_value = params[_tensor_name]
        tensor_value_np = tensor_value.numpy()
        tensor_value_np = tensor_value_np.flatten()
        print(np.size(tensor_value_np))
        tensor_value_np_all = np.append(tensor_value_np_all,tensor_value_np)
        print(np.size(tensor_value_np_all))
        
    bins = np.arange(range_left, range_right, resolution)
    plt.hist(tensor_value_np_all,bins) 
    plt.title("histogram") 
    plt.show()

def plot_quantized_distribution(model_name, tensor_set, range_left, range_right, resolution):
    model = torch.load(model_name)
    print(model)
    params = model.state_dict()
    tensor_value_np_all = np.zeros(1)
    for _tensor_name in tensor_set:
        tensor_value = params[_tensor_name]
        weight_prepack, col_offsets, scale, zero_point = torch.fbgemm_linear_quantize_weight(tensor_value)
        print(weight_prepack)
        print(col_offsets)
        print(scale)
        print(zero_point)
        print(weight_prepack.size())
        tensor_value_np = weight_prepack.numpy()
        tensor_value_np = tensor_value_np.flatten()
        #bins = np.arange(-128, 127, 1)
        #plt.hist(tensor_value_np,bins)
        #plt.show()
        tensor_value_np_all = np.append(tensor_value_np_all,tensor_value_np)
    bins = np.arange(-128, 127, 1)
    plt.hist(tensor_value_np_all,bins) 
    plt.title("histogram") 
    plt.show()

if __name__ == '__main__':
    plot_distribution(MODLE_LOCATION, TENSOR_NAME, -0.5, 0.5, 0.01)
    # pay attention 输入是还没量化的模型
    # plot_quantized_distribution(MODLE_LOCATION, TENSOR_NAME, -128, 127, 1)
    
    

    
