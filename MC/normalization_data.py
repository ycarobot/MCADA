import torch.utils.data as data
from torchvision import transforms
import numpy as np


class myDataset(data.Dataset):
    # rewrite dataset
    def __init__(self, imgs_data, labels, feature_type):
        self.imgs_data = imgs_data
        self.labels = labels
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5],[0.5])])
        if feature_type == 'DeCAF6':
            self.feature = 4096
        elif feature_type == 'Resnet50':
            self.feature = 2048
        elif feature_type == 'MDS':
            self.feature = 400

    def __getitem__(self, index):
        imgs_data = np.asarray(self.imgs_data[index])

        imgs_data = self.transform(imgs_data[:, np.newaxis]).reshape(1, self.feature)
        return imgs_data, self.labels[index]


    def __len__(self):
        return len(self.imgs_data)