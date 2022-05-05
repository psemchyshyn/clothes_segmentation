from dataset import DatasetSeg
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
import albumentations as A


class DataModule(LightningDataModule):
    def __init__(self, path, image_size, batch_size):
        super(DataModule, self).__init__()
        self.transforms = A.Compose([
            A.HorizontalFlip(),
            A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=30, p=0.4),
            A.ElasticTransform(),
            A.GaussNoise(),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(),
                A.RandomGamma(),
                A.MedianBlur(),
            ], p=0.5)
        ])
        self.batch_size = batch_size
        self.image_size = image_size
        # self.ignore_val = ignore_val

        ds = DatasetSeg(path, self.image_size, transforms=self.transforms)
        train_size = int(len(ds)*0.8)
        self.train_ds, self.val_ds = random_split(ds, [train_size, len(ds) - train_size])

        self.channels = ds[0][0].shape[0]
        self.num_classes = ds.num_classes

        self.dims = (self.channels, self.num_classes, self.batch_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, pin_memory=True, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, pin_memory=True, batch_size=self.batch_size)
