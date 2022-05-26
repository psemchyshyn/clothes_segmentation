import yaml
from datamodule import DataModule
from lightning_segmentation import LitSegClothes
import torch
import os
import matplotlib.pyplot as plt

def parse_config(config_path):
    with open(config_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


if __name__ == "__main__":
    conf = parse_config("./config.yaml")
    conf_data = conf["data"]

    dm = DataModule(conf_data)
    model = LitSegClothes.load_from_checkpoint("checkpoints_seg_1class_no_aug/model-epoch=13-val_loss=0.05.ckpt")
    x, mask = next(iter(dm.val_dataloader()))
    y = model(x)
    try:
        os.mkdir("./inference")
    except OSError as error:
        pass

    _, preds = torch.max(y, dim=1)
    for i, img in enumerate(preds[:5]):
        plt.figure(figsize=(11, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(x[i].permute(2, 1, 0))
        plt.title('Original image')
        plt.subplot(1, 3, 2)
        plt.imshow(mask[i].detach().cpu())
        plt.title('Mask')
        plt.subplot(1, 3, 3)
        plt.imshow(img.detach().cpu())
        plt.title('Model prediction')
        plt.savefig(f"inference/{model.hparams['model_name']}{i}.png")