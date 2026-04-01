from typing import cast

import torch
import lightning as L
from omegaconf import DictConfig

from prexsyn.factory import Config, get_chemical_space, get_model, get_detokenizer
from .data_module import SynthesisBatch


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

    def training_step(self, batch: SynthesisBatch, batch_idx: int) -> torch.Tensor:
        token_types, bb_indices, rxn_indices = batch["synthesis"].unbind(-1)
        loss_dict = self.model(
            descriptors=batch["descriptors"],
            token_types=token_types,
            bb_indices=bb_indices,
            rxn_indices=rxn_indices,
        )
        loss = sum_weighted_losses(loss_dict, self.config.training.loss_weights)
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        for k, v in loss_dict.items():
            self.log(f"train/loss_{k}", v, on_step=True, prog_bar=False, logger=True)

        return loss
