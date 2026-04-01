from typing import cast

import torch
import lightning as L
from omegaconf import DictConfig

from prexsyn.factory import Config, get_chemical_space, get_model, get_detokenizer


def sum_weighted_losses(losses: dict[str, torch.Tensor], weights: dict[str, float] | None) -> torch.Tensor:
    loss: torch.Tensor = torch.zeros_like(list(losses.values())[0])
    if weights is None:
        weights = {k: 1.0 for k in losses.keys()}
    for k in losses.keys():
        loss = loss + weights[k] * losses[k]
    return loss


class PrexSynWrapper(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        if not isinstance(config, DictConfig):
            raise ValueError("Config must be a omegaconf.DictConfig instance, duck-typed as Config for convenience.")
        self.save_hyperparameters(config)
        self.model = get_model(config)

    @property
    def config(self) -> Config:
        return cast(Config, DictConfig(self.hparams))

    @property
    def chemical_space(self):
        if not hasattr(self, "_chemical_space"):
            self._chemical_space = get_chemical_space(self.config.chemical_space)
        return self._chemical_space

    @property
    def detokenizer(self):
        if not hasattr(self, "_detokenizer"):
            self._detokenizer = get_detokenizer(self.config, chemspace=self.chemical_space)
        return self._detokenizer

    def configure_optimizers(self):
        optimizers = [torch.optim.AdamW(self.parameters(), lr=self.config.training.optimizer["lr"])]
        schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
        if self.config.training.scheduler is not None:
            schedulers.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizers[0],
                    T_max=self.config.training.scheduler["T_max"],
                    eta_min=self.config.training.scheduler["eta_min"],
                )
            )
        return optimizers, schedulers
