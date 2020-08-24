import os
import time
import torch


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def time_model_evaluation(model, test_data):
    s = time.time()
    output = model.forward(test_data)
    elapsed = time.time() - s
    print('elapsed time (seconds): ', elapsed)
