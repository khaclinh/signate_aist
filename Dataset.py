import os
import tifffile
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

np.seterr(divide='ignore', invalid='ignore')

# create dataset class
class CustomImageDataset(Dataset):
    def __init__(self, labels_file, img_dir, folders: list = None, transform=None, target_transform=None,
                 apply_shuffle: bool = True, ):
        self.img_labels = pd.read_csv(labels_file, sep='\t')
        self.img_dir = img_dir
        self.folders = folders
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def get_label(self, idx):
        return self.img_labels.iloc[idx, 1]

    def labels(self):
        return self.img_labels.iloc[:, 1]

    def __getitem__(self, idx):
        img_path = None
        for i in range(len(self.folders)):
            check_path = os.path.join(self.img_dir, self.folders[i], self.img_labels.iloc[idx, 0])
            if os.path.isfile(check_path):
                img_path = check_path
                break

        image = tifffile.imread(img_path)
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            # normalization
            if image.max() != image.min():
                image = (image - image.min()) / (image.max() - image.min())
            # transform
            image = self.transform(0.5)(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label