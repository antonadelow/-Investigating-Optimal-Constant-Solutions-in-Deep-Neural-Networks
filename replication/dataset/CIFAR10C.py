from torch.utils.data import Dataset
import numpy as np

class CIFAR10C(Dataset):
    def __init__(self, path, corruption_type, corruption_level, transform=None):
        range_start, range_end = corruption_level*10000, (corruption_level+1)*10000
        self.images = np.load(path+corruption_type+".npy")[range_start:range_end]
        self.labels = np.load(path+"labels.npy").squeeze()[range_start:range_end]
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        lb = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, lb

    def __len__(self):
        return len(self.labels)