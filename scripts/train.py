from pathlib import Path

import click
import lightning as L
import torch
from lightning.pytorch import callbacks, loggers, strategies

from prexsyn.factory import Config
from prexsyn.training.data_module import SynthesisDataModule
from prexsyn.training.model_wrapper import PrexSynWrapper


@click.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--devices", type=int, default=1)
@click.option("--max-epochs", type=int, default=1000)
@click.option("--ckpt-path", type=click.Path(exists=True, path_type=Path), default=None)
def main(config_path: Path, devices: int, max_epochs: int, ckpt_path: Path | None):
    config = Config.from_yaml(config_path)
    config_name = config_path.stem

    torch.set_float32_matmul_precision("medium")
    L.seed_everything(config.training.seed, workers=True)

    dm = SynthesisDataModule(config)
    dm.prepare_data()

    model = PrexSynWrapper(config)
    trainer = L.Trainer(
        accelerator="gpu",
        strategy=strategies.DDPStrategy(static_graph=True),
        num_sanity_val_steps=1,
        callbacks=[
            callbacks.ModelCheckpoint(save_last=True, every_n_epochs=25, save_top_k=-1),
            callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        logger=[
            loggers.WandbLogger(project="prexsyn-dev", save_dir="./logs", name=config_name),
        ],
        devices=devices,
        gradient_clip_val=100.0,
        max_epochs=max_epochs,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
