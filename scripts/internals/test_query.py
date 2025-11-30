import hydra
from omegaconf import DictConfig
from rdkit import Chem

from prexsyn.factories.facade import Facade
from prexsyn.queries.planner import QueryPlanner


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    facade = Facade(cfg)
    cond = facade.property_set["ecfp4"].eq(Chem.MolFromSmiles("CCO"))
    print(cond)

    q = (
        facade.property_set["ecfp4"].eq(Chem.MolFromSmiles("CCO"))
        & ~facade.property_set["ecfp4"].eq(Chem.MolFromSmiles("CC"), weight=0.5)
        & facade.property_set["rdkit_descriptors"].eq("amw", 46.07)
        & facade.property_set["rdkit_descriptor_upper_bound"].gt("amw", 100.0)
    )
    print(q)
    planner = QueryPlanner(q)
    print(planner.get_property_reprs())
    print(planner.get_weights())


if __name__ == "__main__":
    main()
