import io
import tempfile
from pathlib import Path
from typing import Any, cast

import pydot
import PIL.Image
import rdkit.Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdDepictor import Compute2DCoords

from prexsyn_engine.chemistry import Molecule, SynthesisNode
from prexsyn_engine.chemspace import Synthesis, ChemicalSpace, PostfixNotationTokenType


def draw_molecule(mol: Molecule) -> PIL.Image.Image:
    rdk_mol = cast(rdkit.Chem.Mol, mol.to_rdkit_mol())
    Compute2DCoords(rdk_mol)

    # Get the bounding box of the molecule
    conf = rdk_mol.GetConformer()
    xs = [conf.GetAtomPosition(i).x for i in range(rdk_mol.GetNumAtoms())]
    ys = [conf.GetAtomPosition(i).y for i in range(rdk_mol.GetNumAtoms())]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Get size according to the bounding box, with some padding
    padding = 5.0
    width = int(max_x - min_x + 2 * padding) * 15
    height = int(max_y - min_y + 2 * padding) * 15

    d2d = Draw.MolDraw2DCairo(width, height)
    opts = d2d.drawOptions()
    opts.setBackgroundColour((1, 1, 1, 0))  # RGBA white with 0 alpha (transparent)
    # opts.bondLineWidth = 1.0
    opts.setAtomPalette({0: (0.0, 0.0, 0.0)})  # Black for all atoms
    opts.minFontSize = 18  # type: ignore[assignment]
    d2d.DrawMolecule(rdk_mol)
    d2d.FinishDrawing()
    img_data = d2d.GetDrawingText()
    return PIL.Image.open(io.BytesIO(img_data))


class SynthesisDrawer:
    def __init__(self):
        super().__init__()
        self.show_intermediate = False
        self.rankdir = "LR"
        self.fontname = "Fira Sans"
        self.dpi = 200

        self.working_dir_handle = tempfile.TemporaryDirectory()
        self.working_dir = Path(self.working_dir_handle.name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.working_dir_handle.cleanup()

    def mol_node(self, node_id: str, mol: Molecule, annots: list[tuple[str, Any] | str]) -> pydot.Node:
        im_path = self.working_dir / f"{node_id}.png"
        draw_molecule(mol).save(im_path)
        label_lines = ["<"]

        label_lines += [
            '<TABLE STYLE="ROUNDED" BORDER="0" CELLBORDER="0" CELLSPACING="5" CELLPADDING="0" BGCOLOR="grey97">',
            '<TR><TD><IMG SRC="' + im_path.as_posix() + '"/></TD></TR>',
        ]
        for line in annots:
            if isinstance(line, str):
                label_lines.append(f"<TR><TD>{line}</TD></TR>")
            else:
                k, v = line
                if v != "" and v is not None:
                    label_lines.append(f"<TR><TD>{k}: {v}</TD></TR>" if k else f"<TR><TD>{v}</TD></TR>")

        label_lines += ["</TABLE>", ">"]

        return pydot.Node(
            node_id,
            shape="plaintext",
            label="".join(label_lines),
            fontsize="11",
            fontname=self.fontname,
        )

    def draw(self, syn: Synthesis, cs: ChemicalSpace, pdf_output: Path | None = None) -> PIL.Image.Image:
        P = pydot.Dot(
            "",
            graph_type="digraph",
            rankdir=self.rankdir,
            fontname=self.fontname,
            fontsize=8,
            dpi=self.dpi,
            bgcolor="transparent",
            nodesep=0.02,
        )

        leaf_nodes: list[pydot.Node | None] = []
        pfn = syn.postfix_notation().tokens()
        for i, token in enumerate(pfn):
            if token.type == PostfixNotationTokenType.BuildingBlock:
                bb = cs.bb_lib()[token.index]
                leaf_nodes.append(self.mol_node(f"bb_{i}", bb.molecule, [bb.identifier]))
            else:
                leaf_nodes.append(None)

        csyn = syn.synthesis()

        queue: list[SynthesisNode] = []
        for i in range(csyn.stack_size()):
            queue.append(csyn.stack_top(i))

        nodes: dict[str, pydot.Node] = {}
        edges: set[tuple[str, str, str]] = set()

        while len(queue) > 0:
            snode = queue.pop(0)
            pfn_node = leaf_nodes[snode.index()]
            if pfn_node is not None:
                node_id = snode.at(0).smiles()
                nodes[node_id] = pfn_node
            else:
                rxn_token = pfn[snode.index()]
                rxn = cs.rxn_lib()[rxn_token.index]
                for product_idx in range(snode.size()):
                    product_mol = snode.at(product_idx)
                    node_id = product_mol.smiles()
                    nodes[node_id] = self.mol_node(f"product_{node_id}", product_mol, [rxn.name])

                    precursors = snode.precursors(product_idx)
                    for precursor in precursors:
                        edges.add(
                            (
                                precursor.molecule.smiles(),
                                node_id,
                                precursor.reactant_name,
                            )
                        )

                for pre_node in snode.precursor_nodes():
                    queue.append(pre_node)

        for node in nodes.values():
            P.add_node(node)

        for src_id, dst_id, label in edges:
            P.add_edge(
                pydot.Edge(
                    nodes[src_id],
                    nodes[dst_id],
                    label=label,
                    fontsize=8,
                    fontname=self.fontname,
                )
            )

        if pdf_output is not None:
            P.write_pdf(pdf_output)  # type: ignore[attr-defined]

        return PIL.Image.open(io.BytesIO(P.create_png()))  # type: ignore[attr-defined]
