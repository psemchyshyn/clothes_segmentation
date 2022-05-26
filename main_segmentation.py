import yaml
from datamodule import DataModule
from lightning_segmentation import LitSegClothes
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def parse_config(config_path):
    with open(config_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


if __name__ == "__main__":
    conf = parse_config("./config.yaml")
    conf_model = conf["model"]
    conf_data = conf["data"]

    dm = DataModule(conf_data)
    lit = LitSegClothes(conf_model["model_name"],
                        conf_model["encoder_name"],
                        conf_model["pretrained_encoder"],
                        conf_data["channels"],
                        conf_data["num_classes"],
                        conf_data["batch_size"],
                        conf_model["lr"])

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{conf_model['save_weights_dir']}",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    estopping_callback = EarlyStopping(monitor="val_loss", patience=2)

    trainer = Trainer(progress_bar_refresh_rate=10,
                      max_epochs=20,
                      check_val_every_n_epoch=1,
                      callbacks=[checkpoint_callback],
                      num_sanity_val_steps=1,
                      default_root_dir=f"{conf_model['save_logs_dir']}",
                      accumulate_grad_batches=1,
                      val_check_interval=0.2,
                      )

    trainer.fit(lit, dm)
