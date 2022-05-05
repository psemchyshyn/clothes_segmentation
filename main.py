import yaml
from datamodule import DataModule
from lightining import LitSegClothes
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def parse_config(config_path):
    with open(config_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


if __name__ == "__main__":
    conf = parse_config("./config.yaml")["data"]
    path = conf["train_path"]
    image_size = conf["image_size"]
    batch_size = conf["batch_size"]
    ignore_val = conf["ignore_val"]
    dm = DataModule(path, image_size, batch_size)

    lit = LitSegClothes(*dm.size())

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    estopping_callback = EarlyStopping(monitor="val_loss", patience=3)

    trainer = Trainer(progress_bar_refresh_rate=10,
                      max_epochs=10,
                      check_val_every_n_epoch=1,
                      callbacks=[estopping_callback, checkpoint_callback],
                      auto_lr_find=True,
                      auto_scale_batch_size=True,
                      num_sanity_val_steps=1,
                      default_root_dir="logs",
                      weights_save_path="checkpoints",
                      accumulate_grad_batches=2
                      )

    trainer.tune(lit, dm)
    trainer.fit(lit, dm)
