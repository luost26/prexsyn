import hashlib
import io
import math
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import PIL.Image
import pydot
import rdkit.Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdDepictor import Compute2DCoords

from prexsyn_engine.chemistry import Molecule
from prexsyn_engine.chemspace import Synthesis

from .syndag import SynDAG


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
    opts.fixedFontSize = 18
    opts.fixedBondLength = 23
    # don't scale the molecule to fit the image, since we already sized it according to the bounding box
    # no need to set: opts.fixedScale = 1
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
        self.bgcolor = "transparent"

    def mol_node(
        self,
        node_id: str,
        mol: Molecule,
        annots: Sequence[tuple[str, Any] | str],
        working_dir: Path,
        highlight: bool = False,
    ) -> pydot.Node:
        im_path = working_dir / f"{node_id}.png"
        draw_molecule(mol).save(im_path)
        label_lines = ["<"]

        border = 2 if highlight else 0
        label_lines += [
            f'<TABLE STYLE="ROUNDED" BORDER="{border}" CELLBORDER="0" CELLSPACING="5" CELLPADDING="0" BGCOLOR="grey97">',
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

    def node_id_from_mol(self, mol: Molecule) -> str:
        return hashlib.md5(mol.smiles().encode()).hexdigest()

    def draw(
        self,
        syn: Synthesis,
        pdf_output: Path | None = None,
        highlight_smiles: str | None = None,
    ) -> PIL.Image.Image:
        with tempfile.TemporaryDirectory() as working_dir_str:
            working_dir = Path(working_dir_str)
            P = pydot.Dot(
                "",
                graph_type="digraph",
                rankdir=self.rankdir,
                fontname=self.fontname,
                fontsize=8,
                dpi=self.dpi,
                bgcolor=self.bgcolor,
                nodesep=0.02,
            )

            dag = SynDAG(syn)

            for node in dag.nodes.values():
                annots: list[str] = []
                if node.building_block is not None:
                    annots.append(node.building_block.identifier)
                if node.reaction is not None:
                    annots.append(node.reaction.name)
                highlight = node.mol.smiles() == highlight_smiles
                P.add_node(self.mol_node(node.key, node.mol, annots, working_dir, highlight))

            for node in dag.nodes.values():
                for prec_dict in node.precursors:
                    for reactant_name, prec_key in prec_dict.items():
                        P.add_edge(
                            pydot.Edge(
                                P.get_node(prec_key)[0],
                                P.get_node(node.key)[0],
                                label=reactant_name,
                                fontsize=8,
                                fontname=self.fontname,
                            )
                        )

            if pdf_output is not None:
                P.write_pdf(pdf_output)  # type: ignore[attr-defined]

            return PIL.Image.open(io.BytesIO(P.create_png()))  # type: ignore[attr-defined]


def make_grid(images: list[PIL.Image.Image]) -> PIL.Image.Image:
    """Make a grid of images.

    Args:
        images (list[PIL.Image.Image]): A list of images.

    Returns:
        PIL.Image.Image: A grid of images.
    """
    width = max(image.size[0] for image in images)
    height = max(image.size[1] for image in images)

    num_cols = int(math.ceil(math.sqrt(len(images))))
    num_rows = int(math.ceil(len(images) / num_cols))
    grid = PIL.Image.new("RGB", (num_cols * width, num_rows * height), color=(255, 255, 255))
    for i, image in enumerate(images):
        x = width * (i % num_cols) + (width - image.size[0]) // 2
        y = height * (i // num_cols) + (height - image.size[1]) // 2
        grid.paste(image, (x, y))
    return grid
