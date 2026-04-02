from pathlib import Path

import torch

import prexsyn_engine
from prexsyn.models.embeddings import DescriptorEmbedderConfig
from prexsyn.models.prexsyn import PrexSyn

from .config import ChemicalSpaceConfig, Config, DataPipelineConfig, DescriptorConfig, FeaturizerConfig
from .descriptor_registry import get_descriptor_constructor


def get_chemical_space(cs_conf: ChemicalSpaceConfig):
    if not cs_conf.cache_path.exists():
        raise FileNotFoundError(f"Chemical space cache not found at: {cs_conf.cache_path}")
    return prexsyn_engine.chemspace.ChemicalSpace.deserialize(cs_conf.cache_path)


def peek_chemical_space(cs_conf: ChemicalSpaceConfig):
    if not cs_conf.cache_path.exists():
        raise FileNotFoundError(f"Chemical space cache not found at: {cs_conf.cache_path}")
    return prexsyn_engine.chemspace.ChemicalSpace.peek(cs_conf.cache_path)


def get_token_def(feat_conf: FeaturizerConfig):
    token_def = prexsyn_engine.descriptor.TokenDef()
    if feat_conf.pad_token is not None:
        token_def.pad = feat_conf.pad_token
    if feat_conf.end_token is not None:
        token_def.end = feat_conf.end_token
    if feat_conf.start_token is not None:
        token_def.start = feat_conf.start_token
    if feat_conf.bb_token is not None:
        token_def.bb = feat_conf.bb_token
    if feat_conf.rxn_token is not None:
        token_def.rxn = feat_conf.rxn_token
    return token_def


def get_descriptor_function(desc_conf: DescriptorConfig):
    constructor = get_descriptor_constructor(desc_conf.type)
    return constructor()


def get_descriptor_embedder_config(desc_conf: DescriptorConfig) -> DescriptorEmbedderConfig:
    fn = get_descriptor_function(desc_conf)
    return {
        "descriptor_dim": fn.num_elements(),
        "num_tokens": desc_conf.num_embedding_tokens,
    }


def get_enumerator_config(dp_conf: DataPipelineConfig):
    conf = prexsyn_engine.enumerator.EnumeratorConfig()
    if dp_conf.heavy_atom_limit is not None:
        conf.heavy_atom_limit = dp_conf.heavy_atom_limit
    if dp_conf.max_building_blocks is not None:
        conf.max_building_blocks = dp_conf.max_building_blocks
    if dp_conf.selectivity_cutoff is not None:
        conf.selectivity_cutoff = dp_conf.selectivity_cutoff
    if dp_conf.max_outcomes_per_reaction is not None:
        conf.max_outcomes_per_reaction = dp_conf.max_outcomes_per_reaction
    return conf


def get_data_pipeline(conf: Config, chemspace: prexsyn_engine.chemspace.ChemicalSpace | None = None):
    if chemspace is None:
        chemspace = get_chemical_space(conf.chemical_space)

    mol_descs = {desc_conf.type: get_descriptor_function(desc_conf) for desc_conf in conf.descriptors}
    token_def = get_token_def(conf.featurizer)

    data_pipeline = prexsyn_engine.datapipe.DataPipeline(
        chemspace,
        mol_descs,
        {
            "synthesis": prexsyn_engine.descriptor.SynthesisPostfixNotation.create(
                token_def=token_def,
                max_length=conf.featurizer.max_length,
            )
        },
    )

    return data_pipeline


def get_model(conf: Config):
    desc_confs: dict[str, DescriptorEmbedderConfig] = {}
    for desc_conf in conf.descriptors:
        desc_embedder_config = get_descriptor_embedder_config(desc_conf)
        desc_confs[desc_conf.type] = desc_embedder_config

    token_def = get_token_def(conf.featurizer)

    cs_stats = peek_chemical_space(conf.chemical_space)

    return PrexSyn(
        dim=conf.model.dim,
        nhead=conf.model.nhead,
        dim_feedforward=conf.model.dim_feedforward,
        num_layers=conf.model.num_layers,
        bb_embed_dim=conf.model.bb_embed_dim,
        descriptor_configs=desc_confs,
        num_token_types=5,
        max_bb_index=cs_stats.num_building_blocks - 1,
        max_rxn_index=cs_stats.num_reactions - 1,
        pad_token=token_def.pad,
        end_token=token_def.end,
        start_token=token_def.start,
        bb_token=token_def.bb,
        rxn_token=token_def.rxn,
    )


def get_detokenizer(
    conf: Config,
    max_outcomes_per_reaction: int = 8,
    chemspace: prexsyn_engine.chemspace.ChemicalSpace | None = None,
):
    if chemspace is None:
        chemspace = get_chemical_space(conf.chemical_space)
    return prexsyn_engine.detokenizer.MultiThreadedDetokenizer(
        chemical_space=chemspace,
        token_def=get_token_def(conf.featurizer),
        max_outcomes_per_reaction=max_outcomes_per_reaction,
    )


def load_model_and_config(model_path: Path) -> tuple[PrexSyn, Config]:
    conf_path = model_path.with_suffix(".yml")
    if not conf_path.exists():
        raise FileNotFoundError(f"Config file not found for model at {model_path}, expected at {conf_path}")

    config = Config.from_yaml(conf_path)
    model = get_model(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model, config
