from typing import cast

import lightning as L
import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from prexsyn.factory import Config, get_chemical_space, get_descriptor_constructor, get_detokenizer, get_model
from prexsyn.samplers.basic import BasicSampler
from prexsyn.utils.draw import SynthesisDrawer, make_grid
from prexsyn.utils.metrics import tanimoto_similarity

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

    @property
    def fingerprint_function(self):
        if not hasattr(self, "_fingerprint_function"):
            self._fingerprint_function = get_descriptor_constructor("ecfp4")()
        return self._fingerprint_function

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

    def validation_step(self, batch: SynthesisBatch, batch_idx: int) -> torch.Tensor:
        token_types, bb_indices, rxn_indices = batch["synthesis"].unbind(-1)
        loss_dict = self.model(
            descriptors=batch["descriptors"],
            token_types=token_types,
            bb_indices=bb_indices,
            rxn_indices=rxn_indices,
        )
        loss = sum_weighted_losses(loss_dict, self.config.training.loss_weights)
        self.log("val/loss", loss, on_step=False, prog_bar=True, logger=True)
        for k, v in loss_dict.items():
            self.log(f"val/loss_{k}", v, on_step=False, prog_bar=False, logger=True)

        sampler = BasicSampler(self.model, num_samples=1)
        syn_pred_list = sampler.sample(batch["descriptors"]).detokenize(self.detokenizer)
        syn_true_list = self.detokenizer(batch["synthesis"].cpu().numpy())
        sim_list: list[float] = []
        count_success = 0
        for syn_true, syn_pred in zip(syn_true_list, syn_pred_list):
            if syn_pred.synthesis().stack_size() != 1:
                sim_list.append(0.0)
                continue

            prod_pred = syn_pred.products()[0]
            prod_true = syn_true.products()[0]
            fp_pred = self.fingerprint_function(prod_pred)
            fp_true = self.fingerprint_function(prod_true)
            sim_list.append(float(tanimoto_similarity(fp_true, fp_pred)))
            count_success += 1
        self.log("val/similarity", float(np.mean(sim_list)), on_step=False, prog_bar=True, logger=True)
        self.log("val/success_rate", count_success / len(syn_pred_list), on_step=False, prog_bar=False, logger=True)

        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            with SynthesisDrawer() as drawer:
                drawer.bgcolor = "white"
                images = [
                    make_grid([drawer.draw(syn_true), drawer.draw(syn_pred)])
                    for syn_true, syn_pred in zip(syn_true_list[:50], syn_pred_list[:50])
                ]
                self.logger.log_image(
                    key="val/samples",
                    images=images,
                )

        return loss
