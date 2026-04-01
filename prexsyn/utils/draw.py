import io
from typing import cast

import PIL.Image
import rdkit.Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdDepictor import Compute2DCoords

from prexsyn_engine.chemistry import Molecule


def draw_molecule(mol: Molecule, size: int | tuple[int, int] = 300, recompute_coords: bool = False) -> PIL.Image.Image:
    rdk_mol = cast(rdkit.Chem.Mol, mol.to_rdkit_mol())
    if recompute_coords:
        Compute2DCoords(rdk_mol)
    size = (size, size) if isinstance(size, int) else size

    d2d = Draw.MolDraw2DCairo(size[0], size[1])
    opts = d2d.drawOptions()
    opts.setBackgroundColour((1, 1, 1, 0))  # RGBA white with 0 alpha (transparent)
    # opts.bondLineWidth = 1.0
    opts.setAtomPalette({0: (0.0, 0.0, 0.0)})  # Black for all atoms
    opts.minFontSize = 20  # type: ignore[assignment]
    d2d.DrawMolecule(rdk_mol)
    d2d.FinishDrawing()
    img_data = d2d.GetDrawingText()
    return PIL.Image.open(io.BytesIO(img_data))
