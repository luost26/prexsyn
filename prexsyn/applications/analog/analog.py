import time

from rdkit import Chem

from prexsyn.data.struct import move_to_device
from prexsyn.factories.property import BasePropertyDef
from prexsyn.models.prexsyn import PrexSyn
from prexsyn.samplers.basic import BasicSampler
from prexsyn_engine.detokenizer import Detokenizer
from prexsyn_engine.fingerprints import mol_to_syntheses_tanimoto_similarity

from .data import AnalogGenerationResult


def generate_analogs(
    model: PrexSyn,
    sampler: BasicSampler,
    detokenizer: Detokenizer,
    fp_property: BasePropertyDef,
    mol: Chem.Mol,
    eval_fp_type: str = "ecfp4",
) -> AnalogGenerationResult:
    t_start = time.perf_counter()
    property_repr = {fp_property.name: move_to_device(fp_property.evaluate_mol(mol), model.device)}
    synthesis_repr = sampler.sample(property_repr)
    syn_list = detokenizer(
        token_types=synthesis_repr["token_types"].cpu().numpy(),
        bb_indices=synthesis_repr["bb_indices"].cpu().numpy(),
        rxn_indices=synthesis_repr["rxn_indices"].cpu().numpy(),
    )
    sim_matrix = mol_to_syntheses_tanimoto_similarity(mol, syn_list, fp_type=eval_fp_type)
    sim_list = sim_matrix.max(axis=1)
    max_sim_product_idx = sim_matrix.argmax(axis=1)
    t_end = time.perf_counter()
    return {
        "synthesis": list(syn_list),
        "similarity": sim_list,
        "max_sim_product_idx": max_sim_product_idx,
        "time_taken": t_end - t_start,
    }
