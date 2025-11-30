from prexsyn.factories.chemical_space import ChemicalSpace
from prexsyn.factories.tokenization import Tokenization
from prexsyn.properties import PropertySet
from prexsyn.properties.fingerprint import ECFP4, FCFP4
from prexsyn.properties.rdkit_descriptors import RDKitDescriptors
from prexsyn.training.online_dataset import OnlineSynthesisDataset

if __name__ == "__main__":
    cs = ChemicalSpace.load_or_create(
        building_block_path="data/building_blocks/Enamine_Rush-Delivery_Building_Blocks-US_223244cmpd_20231001.sdf",
        reaction_path="data/reactions/rxn115.txt",
        cache_dir="data/chemical_spaces/enamine_rxn115",
    )
    print(cs.get_csd())
    print(cs.count_building_blocks())
    print(cs.count_reactions())

    pset = PropertySet([ECFP4(), RDKitDescriptors()])
    pset.add(FCFP4())

    dataset = OnlineSynthesisDataset(
        chemical_space=cs,
        property_set=pset,
        tokenization=Tokenization(),
    )
    for data in dataset:
        print(data)
        break
