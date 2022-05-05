from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import pandas as pd
import numpy as np


class DatasetSeg(Dataset):
    def __init__(self, path, image_size, transforms=None):
        self.image_size = image_size
        self.path = os.path.normpath(path)
        # self.ignore_val = ignore_val

        self.image_path = os.path.join(self.path, "train")
        self.image_list = os.listdir(self.image_path)

        self.df = pd.read_csv(os.path.join(self.path, "train.csv"))
        self.transforms = transforms
        self.df['CategoryId'] = self.df['ClassId'].str.split('_').str[0]
        self.df['AttributeId'] = self.df['ClassId'].str.split('_').str[1:]

        self.num_classes = len(self.df["CategoryId"].unique())

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_path, self.image_list[idx])).convert("RGB")
        mask = self.get_mask(idx)

        img = np.array(img.resize((self.image_size, self.image_size)))
        mask = np.array(mask.resize((self.image_size, self.image_size)))
        mask[mask >= self.num_classes] = self.num_classes
        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(mask)

    def get_mask(self, img_id):
        '''
        https://www.kaggle.com/code/josutk/show-segmentation
        :param img_id:
        :return:
        '''
        img_id = self.image_list[img_id]
        a = self.df[self.df.ImageId == img_id]
        a = a.groupby('CategoryId', as_index=False).agg({'EncodedPixels': ' '.join, 'Height': 'first', 'Width': 'first'})
        H = a.iloc[0, 2]
        W = a.iloc[0, 3]
        mask = np.full(H * W, dtype='int', fill_value=self.num_classes)
        for line in a[['EncodedPixels', 'CategoryId']].iterrows():
            encoded = line[1][0]
            pixel_loc = list(map(int, encoded.split(' ')[0::2]))
            iter_num = list(map(int, encoded.split(' ')[1::2]))
            for p, i in zip(pixel_loc, iter_num):
                mask[p: (p + i)] = line[1][1]
        mask = mask.reshape(W, H).T
        mask = Image.fromarray(np.uint8(mask)).convert('L')
        return mask
