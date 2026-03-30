import torch
from torch.utils.data import Dataset


class HatefulMemesDataset(Dataset):
    def __init__(self, features_dict):
        self.img_features = features_dict['img_features']
        self.txt_features = features_dict['txt_features']
        self.labels = features_dict['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'img_features': self.img_features[idx],
            'txt_features': self.txt_features[idx],
            'label': self.labels[idx],
        }
