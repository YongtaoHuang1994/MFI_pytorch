# 训练和测试数据生成器

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
import cv2

class MfiDataset(Dataset):

    def __init__(self, root_dir, names_file, transform):
        self.root_dir = root_dir
        self.names_file = names_file
        self.size = 0
        self.names_list = []
        self.transform = transform

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        file = open(self.names_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.root_dir + self.names_list[idx].split(',')[0]
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        label = int(self.names_list[idx].split(',')[1])
        sample = [image, label]
        #print(sample)
        return sample
