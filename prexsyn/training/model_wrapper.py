from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from prexsyn.data.struct import SynthesisTrainingBatch
from prexsyn.factories.facade import Facade
from prexsyn.samplers.basic import BasicSampler
from prexsyn.utils.draw import draw_synthesis, make_grid
from prexsyn_engine.fingerprints import tanimoto_similarity


def sum_weighted_losses(losses: dict[str, torch.Tensor], weights: dict[str, float] | None) -> torch.Tensor:
    loss: torch.Tensor = torch.zeros_like(list(losses.values())[0])
    if weights is None:
        weights = {k: 1.0 for k in losses.keys()}
    for k in losses.keys():
        loss = loss + weights[k] * losses[k]
    return loss


class PrexSynWrapper(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.facade = Facade(cfg)
        self.model_dim: int = cfg.model.dim
        self.model = self.facade.build_model()

    def configure_optimizers(self) -> Any:
        optimizers = [torch.optim.AdamW(self.parameters(), lr=self.hparams["training"]["optimizer"]["lr"])]
        schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
        if "scheduler" in self.hparams["training"]:
            schedulers.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizers[0],
                    T_max=self.hparams["training"]["scheduler"]["T_max"],
                    eta_min=self.hparams["training"]["scheduler"]["eta_min"],
                )
            )
        return optimizers, schedulers

    def training_step(self, batch: SynthesisTrainingBatch, batch_idx: int) -> torch.Tensor:
        loss_dict = self.model(property_repr=batch["property_repr"], synthesis_repr=batch["synthesis_repr"])
        loss = sum_weighted_losses(loss_dict, self.hparams["model"]["loss_weights"])
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        for k, v in loss_dict.items():
            self.log(f"train/loss_{k}", v, on_step=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch: SynthesisTrainingBatch, batch_idx: int) -> torch.Tensor:
        loss_dict = self.model(
            property_repr=batch["property_repr"],
            synthesis_repr=batch["synthesis_repr"],
        )

        loss = sum_weighted_losses(loss_dict, self.hparams["model"]["loss_weights"])
        self.log("val/loss", loss, on_step=False, prog_bar=True, logger=True)
        for k, v in loss_dict.items():
            self.log(f"val/loss_{k}", v, on_step=False, prog_bar=False, logger=True)

        samples = BasicSampler(self.model, token_def=self.facade.tokenization.token_def).sample(batch["property_repr"])
        syn_true_list = self.facade.get_detokenizer()(
            token_types=batch["synthesis_repr"]["token_types"].cpu().numpy(),
            bb_indices=batch["synthesis_repr"]["bb_indices"].cpu().numpy(),
            rxn_indices=batch["synthesis_repr"]["rxn_indices"].cpu().numpy(),
        )
        syn_pred_list = self.facade.get_detokenizer()(
            token_types=samples["token_types"].cpu().numpy(),
            bb_indices=samples["bb_indices"].cpu().numpy(),
            rxn_indices=samples["rxn_indices"].cpu().numpy(),
        )
        sim_list: list[float] = []
        count_success = 0
        for syn_true, syn_pred in zip(syn_true_list, syn_pred_list):
            if syn_pred.stack_size() != 1:
                sim_list.append(0.0)
                continue
            sim_list.append(
                tanimoto_similarity(
                    syn_true.top().to_list()[0],
                    syn_pred.top().to_list()[0],
                    fp_type="ecfp4",
                )
            )
            count_success += 1

        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            images = [
                make_grid([draw_synthesis(syn_true), draw_synthesis(syn_pred)])
                for syn_true, syn_pred in zip(syn_true_list[:50], syn_pred_list[:50])
            ]
            self.logger.log_image(
                key="val/samples",
                images=images,
            )

        self.log("val/similarity", float(np.mean(sim_list)), on_step=False, prog_bar=True, logger=True)
        self.log("val/success_rate", count_success / len(syn_pred_list), on_step=False, prog_bar=False, logger=True)

        return loss
