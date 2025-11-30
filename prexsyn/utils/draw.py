import io
import math
import tempfile
from collections.abc import Mapping
from typing import Any

import PIL.Image
import pydot
from rdkit.Chem import Draw
from rdkit.Chem.rdDepictor import Compute2DCoords

from prexsyn_engine.synthesis import Synthesis
from prexsyn_engine.types import Mol, Reaction


def draw_molecule(mol: Mol, size: int | tuple[int, int] = 300, recompute_coords: bool = False) -> PIL.Image.Image:
    if recompute_coords:
        Compute2DCoords(mol)
    size = (size, size) if isinstance(size, int) else size

    d2d = Draw.MolDraw2DCairo(size[0], size[1])
    opts = d2d.drawOptions()
    opts.setBackgroundColour((1, 1, 1, 0))  # RGBA white with 0 alpha (transparent)
    # opts.bondLineWidth = 1.0
    opts.setAtomPalette({0: (0.0, 0.0, 0.0)})  # Black for all atoms
    opts.minFontSize = 20  # type: ignore[assignment]
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    img_data = d2d.GetDrawingText()
    return PIL.Image.open(io.BytesIO(img_data))


def draw_reaction(rxn: Reaction, size: int | tuple[int, int] = 300) -> PIL.Image.Image:
    size = (size, size) if isinstance(size, int) else size
    return Draw.ReactionToImage(rxn, subImgSize=size)  # type: ignore[no-untyped-call,no-any-return]


def _get_annots(o: Mol | Reaction, keys: Mapping[str, str]) -> dict[str, str]:
    annots: dict[str, str] = {}
    for name, key in keys.items():
        if o.HasProp(key):
            annots[name] = o.GetProp(key)
        else:
            annots[name] = ""
    return annots


def draw_synthesis(
    s: Synthesis,
    node_image_size: int = 200,
    show_intermediate: bool = False,
    show_num_cases: bool = False,
    product: Mol | None = None,
    rankdir: str = "LR",
    fontname: str = "Fira Sans",
    dpi: int = 200,
    show_step: bool = False,
    pdf_output: str | None = None,
) -> PIL.Image.Image:
    with tempfile.TemporaryDirectory() as tmpdir:

        def _make_node(node_name: str, mol: Mol | None, annots: dict[str, Any]) -> pydot.Node:
            im_path = f"{tmpdir}/{node_name}.png"
            if mol is not None:
                draw_molecule(mol, size=node_image_size).save(im_path)
            label_lines = [
                "<",
                '<TABLE STYLE="ROUNDED" BORDER="0" CELLBORDER="0" CELLSPACING="5" CELLPADDING="0" BGCOLOR="grey97">',
                ('<TR><TD><IMG SRC="' + im_path + '"/></TD></TR>' if mol is not None else ""),
            ]
            for k, v in annots.items():
                if v != "" and v is not None:
                    label_lines.append(f"<TR><TD>{k}: {v}</TD></TR>" if k else f"<TR><TD>{v}</TD></TR>")
            label_lines += ["</TABLE>", ">"]

            return pydot.Node(
                node_name, shape="plaintext", label="".join(label_lines), fontsize="11", fontname=fontname
            )

        P = pydot.Dot(
            "",
            graph_type="digraph",
            rankdir=rankdir,
            fontname=fontname,
            fontsize=8,
            dpi=dpi,
            bgcolor="transparent",
            nodesep=0.02,
        )
        node_stack: list[str] = []
        replay = Synthesis()
        pfn_list = s.get_postfix_notation().to_list()
        for i, item in enumerate(pfn_list):
            node_name = str(i)
            if isinstance(item, Mol):
                node_stack.append(node_name)
                replay.push_mol(item)
                P.add_node(
                    _make_node(
                        node_name,
                        item,
                        {
                            "Step": i if show_step else None,
                            **_get_annots(
                                item,
                                {"": "building_block_index", "Name": "name"},
                            ),
                        },
                    )
                )
            elif isinstance(item, Reaction):
                replay.push_reaction(item)
                if show_intermediate:
                    prod = replay.top().to_list()[0]
                    Compute2DCoords(prod)
                elif product is not None and i == len(pfn_list) - 1:
                    prod = product
                    Compute2DCoords(prod)
                else:
                    prod = None
                P.add_node(
                    _make_node(
                        node_name,
                        prod,
                        {
                            "Step": i if show_step else None,
                            **_get_annots(
                                item,
                                {"Reaction": "reaction_index", "Name": "name"},
                            ),
                            "#Cases": len(replay.top()) if show_num_cases else None,
                        },
                    )
                )
                for _ in range(item.GetNumReactantTemplates()):
                    P.add_edge(pydot.Edge(node_stack.pop(), node_name, color="grey50"))
                node_stack.append(node_name)
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
