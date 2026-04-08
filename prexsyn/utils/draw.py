import io
import math
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pydot
import rdkit.Chem
from matplotlib import cm
from rdkit.Chem import Draw
from rdkit.Chem.rdDepictor import Compute2DCoords

from prexsyn_engine.chemistry import Molecule
from prexsyn_engine.chemspace import Synthesis

from .syndag import SynthesisDAG


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


class SynthesisDraw:
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

    def draw(
        self,
        syn: Synthesis,
        pdf_output: Path | None = None,
        highlight_smiles: str | None = None,
    ) -> PIL.Image.Image:
        node_key_to_id: dict[str, str] = {}

        def _get_node_id(node_key: str) -> str:
            if node_key not in node_key_to_id:
                node_id = f"{len(node_key_to_id)}"
                node_key_to_id[node_key] = node_id
            return node_key_to_id[node_key]

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

            dag = SynthesisDAG(syn)

            for node in dag.nodes.values():
                annots: list[str] = []
                if node.building_block is not None:
                    annots.append(node.building_block.identifier)
                if node.reaction is not None:
                    annots.append(node.reaction.name)
                highlight = node.mol.smiles() == highlight_smiles
                P.add_node(self.mol_node(_get_node_id(node.key), node.mol, annots, working_dir, highlight))

            for node in dag.nodes.values():
                for prec_dict in node.precursors:
                    for reactant_name, prec_key in prec_dict.items():
                        P.add_edge(
                            pydot.Edge(
                                P.get_node(_get_node_id(prec_key))[0],
                                P.get_node(_get_node_id(node.key))[0],
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


def draw_fingerprint(fp: np.ndarray, grid_shape: tuple[int, int] = (8, 48), dpi: int = 200) -> PIL.Image.Image:
    """Visualize a binary fingerprint on a user-specified 2D grid."""
    bits = np.asarray(fp).astype(bool).ravel()
    rows, cols = grid_shape
    if rows <= 0 or cols <= 0:
        raise ValueError("grid_shape must contain positive rows and cols")
    n_grid_cells = rows * cols

    n_shades_per_side = 8
    cmap = cm.get_cmap("tab10", n_shades_per_side * 2)
    desat_mix = 0.28
    neutral = np.array([0.90, 0.90, 0.90], dtype=float)

    color_sum = np.zeros((n_grid_cells, 3), dtype=float)
    color_count = np.zeros(n_grid_cells, dtype=int)
    occupied = np.zeros(n_grid_cells, dtype=bool)

    neighbor_offsets = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    for idx in np.flatnonzero(bits):
        mapped_idx = int(idx % n_grid_cells)
        mapped_r, mapped_c = divmod(mapped_idx, cols)

        placed_idx = mapped_idx
        if occupied[mapped_idx]:
            for dr, dc in neighbor_offsets:
                rr = mapped_r + dr
                cc = mapped_c + dc
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    continue
                candidate_idx = rr * cols + cc
                if not occupied[candidate_idx]:
                    placed_idx = candidate_idx
                    break

        # Color is strictly a function of the original bit dimension (idx).
        h = ((idx * 2654435761) & 0xFFFFFFFF) / 0xFFFFFFFF
        palette_bins = n_shades_per_side * 2
        palette_idx = min(int(h * palette_bins), palette_bins - 1)

        base = np.array(cmap(palette_idx)[:3], dtype=float)
        bit_color = (1.0 - desat_mix) * base + desat_mix * neutral
        color_sum[placed_idx] += bit_color
        color_count[placed_idx] += 1
        occupied[placed_idx] = True

    active_mask = color_count > 0
    active_idx = np.flatnonzero(active_mask)
    rgb_flat = np.ones((n_grid_cells, 3), dtype=float)
    rgb_flat[active_idx] = color_sum[active_idx] / color_count[active_idx, None]

    fig_w = max(3.4, cols * 0.095)
    fig_h = max(2.6, rows * 0.095)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    # Draw all valid bit positions as faint circles to keep structure visible.
    all_idx = np.arange(n_grid_cells)
    all_r = all_idx // cols
    all_c = all_idx % cols
    ax.scatter(all_c, all_r, s=18, c="#F1F1F1", marker="o", linewidths=0)

    # Draw active bits as colored circles with a soft halo for tiny previews.
    active_r = active_idx // cols
    active_c = active_idx % cols
    active_colors = rgb_flat[active_idx]
    density = np.clip(color_count[active_idx], 1, 6)
    halo_sizes = 95 + 18 * density
    core_sizes = 40 + 10 * density
    ax.scatter(active_c, active_r, s=halo_sizes, c=active_colors, marker="o", alpha=0.24, linewidths=0)
    ax.scatter(active_c, active_r, s=core_sizes, c=active_colors, marker="o", alpha=0.98, linewidths=0)

    ax.set_xlim(-0.6, cols - 0.4)
    ax.set_ylim(rows - 0.4, -0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return PIL.Image.open(buf)
