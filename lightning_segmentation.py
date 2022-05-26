import torch
import pytorch_lightning as pl
import torchmetrics
import os
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp


class LitSegClothes(pl.LightningModule):
    def __init__(self, model_name, encoder_name, pretrained_encoder, channels, num_classes, batch_size=32, lr=1e-3):
        super(LitSegClothes, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.pretrained_encoder = "imagenet" if pretrained_encoder else None
        self.save_hyperparameters()

        self.model = smp.create_model(model_name, encoder_name, encoder_weights=self.pretrained_encoder, in_channels=channels, classes=num_classes + 1) # for background class

        self.loss_fn = smp.losses.DiceLoss(smp.losses.constants.MULTICLASS_MODE, from_logits=True)

        params = smp.encoders.get_preprocessing_params(self.encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(), torchmetrics.JaccardIndex(self.num_classes + 1), torchmetrics.F1Score(mdmc_average="global")])
        self.metrics = {"train": metrics.clone("train"), "val": metrics.clone("val")}


    def forward(self, x):
        x = (x - self.mean) / self.std
        mask = self.model(x)
        return mask

    def step(self, batch, batch_idx, type):
        x, mask = batch
        mask = mask.long()
        y = self(x)

        loss = self.loss_fn(y, mask)
        self.log_dict({f"{type}_loss": loss, **self.metrics[type](y, mask)})
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [sch]

    def on_validation_epoch_end(self):
        x, mask = next(iter(self.trainer.val_dataloaders[0]))
        y = self(x)

        try:
            os.mkdir("./examples")
        except OSError as error:
            pass

        _, preds = torch.max(y, dim=1)
        for i, img in enumerate(preds[:5]):
            plt.figure(figsize=(11,4))
            plt.subplot(1, 3, 1)
            plt.imshow(x[i].permute(2, 1, 0))
            plt.title('Original image')
            plt.subplot(1, 3, 2)
            plt.imshow(mask[i].detach().cpu())
            plt.title('Mask')
            plt.subplot(1, 3, 3)
            plt.imshow(img.detach().cpu())
            plt.title('Model prediction')
            plt.savefig(f"examples/{self.trainer.current_epoch}-{i}.png")
