from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import pandas as pd
import numpy as np


class DatasetSeg(Dataset):
    def __init__(self, path, image_size, num_classes, class_size, transforms=None):
        self.image_size = image_size
        self.path = os.path.normpath(path)

        self.image_path = os.path.join(self.path, "train")
        self.image_list = os.listdir(self.image_path)

        self.df = pd.read_csv(os.path.join(self.path, "train.csv"))
        self.transforms = transforms

        self.df['CategoryId'] = self.df['ClassId'].str.split('_').str[0].astype(int)
        self.df["Height"] = self.df["Height"].astype(int)
        self.df["Width"] = self.df["Width"].astype(int)

        # unite classes
        self.df['CategoryId'] = self.df['CategoryId'] % num_classes
        # reduce dataset
        self.df = self.df.groupby("CategoryId").head(class_size)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx, 0]
        img = Image.open(os.path.join(self.image_path, image_id)).convert("RGB")
        mask = self.get_mask(image_id)

        img = np.array(img.resize((self.image_size, self.image_size)))
        mask = np.array(mask.resize((self.image_size, self.image_size)))
        mask[mask >= self.num_classes] = self.num_classes
        if self.transforms:
            transformed = self.transforms(image=img, masked=mask)
            img = transformed["image"]
            mask = transformed["masked"]
        return torch.from_numpy(img).permute(2, 1, 0) / 255, torch.from_numpy(mask)

    def get_mask(self, img_id):
        '''
        https://www.kaggle.com/code/josutk/show-segmentation
        :param img_id:
        :return:
        '''
        a = self.df[self.df.ImageId == img_id]
        a = a.groupby('CategoryId', as_index=False).agg({'EncodedPixels': ' '.join, 'Height': 'first', 'Width': 'first'})
        H = int(a.iloc[0, 2])
        W = int(a.iloc[0, 3])
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

# import matplotlib.pylab as plt
# import albumentations as A
#
# transforms = A.Compose([
#     A.HorizontalFlip(),
#     A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=30, p=0.4),
#     A.ElasticTransform(),
#     A.GaussNoise(),
#     A.OneOf([
#         A.CLAHE(clip_limit=2),
#         A.RandomBrightnessContrast(),
#         A.RandomGamma(),
#         A.MedianBlur(),
#     ], p=0.5)
# ])
# path = "E:\\Users\\msemc\\Documents\\Pavlo\\DATA\\imaterialist-fashion-2019-FGVC6"
# ds = DatasetSeg(path, 128, 7, 1000, transforms)
#
# x, mask = ds[0]
# plt.figure(figsize=(11, 4))
# plt.subplot(1, 3, 1)
# plt.imshow(x.permute(2, 1, 0))
# plt.title('Original image')
# plt.subplot(1, 3, 2)
# plt.imshow(mask)
# plt.title('Mask')
# plt.savefig(f"hello.png")