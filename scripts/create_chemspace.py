from pathlib import Path

import click

import prexsyn_engine
from prexsyn.factory import Config


def cached_load_building_blocks(bb_path: Path):
    cache_path = bb_path.with_suffix(".cache")
    if cache_path.exists():
        print(f"Loading building blocks from cache: {cache_path}")
        return prexsyn_engine.chemspace.BuildingBlockLibrary.deserialize(cache_path)
    else:
        bb_lib = prexsyn_engine.chemspace.bb_lib_from_sdf(bb_path)
        bb_lib.serialize(cache_path)
        return bb_lib


def cached_load_reactions(rxn_path: Path):
    cache_path = rxn_path.with_suffix(".cache")
    if cache_path.exists():
        print(f"Loading reactions from cache: {cache_path}")
        return prexsyn_engine.chemspace.ReactionLibrary.deserialize(cache_path)
    else:
        if rxn_path.suffix == ".csv":
            rxn_lib = prexsyn_engine.chemspace.rxn_lib_from_csv(rxn_path)
        elif rxn_path.suffix == ".txt":
            rxn_lib = prexsyn_engine.chemspace.rxn_lib_from_plain_text(rxn_path)
        else:
            raise ValueError(f"Unsupported reaction library format: {rxn_path.suffix}")
        rxn_lib.serialize(cache_path)
        return rxn_lib


@click.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path), required=True)
def main(config_path: Path):
    config = Config.from_yaml(config_path)
    cs_conf = config.chemical_space
    if cs_conf.cache_path.exists():
        click.confirm(f"Cache file {cs_conf.cache_path} already exists. Do you want to overwrite it?", abort=True)

    if cs_conf.bb_path is None:
        raise ValueError("Building block path (bb_path) must be provided in the config.")
    bb_lib = cached_load_building_blocks(cs_conf.bb_path)

    if cs_conf.rxn_path is None:
        raise ValueError("Reaction path (rxn_path) must be provided in the config.")
    rxn_lib = cached_load_reactions(cs_conf.rxn_path)

    matching_config = prexsyn_engine.chemspace.ReactantMatchingConfig()
    if cs_conf.building_block_selectivity_cutoff is not None:
        matching_config.selectivity_cutoff = cs_conf.building_block_selectivity_cutoff

    cs = prexsyn_engine.chemspace.ChemicalSpace(
        bb_lib=bb_lib,
        rxn_lib=rxn_lib,
        int_lib=prexsyn_engine.chemspace.IntermediateLibrary(),
        matching_config=matching_config,
    )
    cs.build_reactant_lists_for_building_blocks()
    cs.generate_intermediates()
    cs.build_reactant_lists_for_intermediates()
    cs.serialize(cs_conf.cache_path)
    print(f"Chemical space created and cached at {cs_conf.cache_path}")


if __name__ == "__main__":
    main()
