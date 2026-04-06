from dataclasses import dataclass, field
from typing import TypeAlias

from prexsyn_engine.chemistry import Molecule, SynthesisNode
from prexsyn_engine.chemspace import (
    BuildingBlockItem,
    PostfixNotationTokenType,
    ReactionItem,
    Synthesis,
)

_KeyString: TypeAlias = str
_ReactantName: TypeAlias = str


@dataclass
class SynDAGNode:
    key: _KeyString
    mol: Molecule
    building_block: BuildingBlockItem | None = None
    reaction: ReactionItem | None = None
    precursors: list[dict[_ReactantName, _KeyString]] = field(default_factory=list)
    successors: set[tuple[_ReactantName, _KeyString]] = field(default_factory=set)


class SynDAG:
    def __init__(self, syn: Synthesis | None = None):
        super().__init__()
        self.nodes: dict[str, SynDAGNode] = {}
        if syn is not None:
            self.build(syn)

    def get_node(self, mol: Molecule) -> SynDAGNode:
        node_key = mol.smiles()
        if node_key not in self.nodes:
            self.nodes[node_key] = SynDAGNode(node_key, mol)
        return self.nodes[node_key]

    def build(self, syn: Synthesis):
        cs = syn.chemical_space()
        csyn = syn.synthesis()

        queue: list[SynthesisNode] = []
        for i in range(csyn.stack_size()):
            queue.append(csyn.stack_top(i))

        while len(queue) > 0:
            snode = queue.pop(0)
            pfn_token = syn.postfix_notation().tokens()[snode.index()]
            is_bb = pfn_token.type == PostfixNotationTokenType.BuildingBlock

            for prod_idx in range(snode.size()):
                mol = snode.at(prod_idx)
                node = self.get_node(mol)
                if is_bb:
                    node.building_block = cs.bb_lib()[pfn_token.index]
                else:
                    node.reaction = cs.rxn_lib()[pfn_token.index]

                    precursors = snode.precursors(prod_idx)
                    prec_dict: dict[_ReactantName, _KeyString] = {}
                    for precursor in precursors:
                        prec_dict[precursor.reactant_name] = precursor.molecule.smiles()
                        self.get_node(precursor.molecule).successors.add((precursor.reactant_name, node.key))
                    node.precursors.append(prec_dict)

            for prec_node in snode.precursor_nodes():
                queue.append(prec_node)

    def products(self):
        return [node for node in self.nodes.values() if len(node.successors) == 0]

    def to_dict(self, node_key: str):
        node = self.nodes[node_key]
        d: dict[str, object] = {"SMILES": node.mol.smiles()}
        if node.building_block is not None:
            d["BuildingBlock"] = node.building_block.identifier
        if node.reaction is not None:
            d["Reaction"] = node.reaction.name
        if len(node.precursors) > 0:
            d["Precursors"] = []
            for prec_dict in node.precursors:
                prec_entry: dict[str, object] = {}
                for reactant_name, prec_key in prec_dict.items():
                    prec_entry[reactant_name] = self.to_dict(prec_key)
                d["Precursors"].append(prec_entry)
        return d
