import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from functools import partial
from unet.model import UNet


class LitSegClothes(pl.LightningModule):
    def __init__(self, channels, num_classes, batch_size=32, lr=1e-3):
        super(LitSegClothes, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size

        self.model = UNet(self.channels, self.num_classes + 1)

        self.metrics = {"acc": torchmetrics.functional.accuracy,
                        "jaccard_index": partial(torchmetrics.functional.jaccard_index, num_classes=self.num_classes + 1),
                        "dice_coef": torchmetrics.functional.dice_score
                        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, mask = batch
        x = x.float()
        mask = mask.long()
        y = self(x)
        loss = F.cross_entropy(y, mask, ignore_index=self.num_classes)

        for key, metric in self.metrics.items():
            self.log(f"train_{key}", metric(y, mask))
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, mask = batch
        x = x.float()
        mask = mask.long()

        y = self(x)
        loss = F.cross_entropy(y, mask, ignore_index=self.num_classes)
        self.on_train_epoch_end()
        for key, metric in self.metrics.items():
            self.log(f"val_{key}", metric(y, mask))
        self.log("val_loss", loss)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [sch]

    def on_train_epoch_end(self):
        x, _ = next(iter(self.trainer.val_dataloaders[0]))
        x = x.float()

        y = self(x)

        _, preds = torch.max(y, dim=1)
        for i, img in enumerate(preds):
            plt.imsave(f"examples/{self.trainer.current_epoch}-{i}.png", img.numpy())
