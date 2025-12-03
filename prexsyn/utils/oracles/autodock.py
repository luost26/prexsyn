import dataclasses
import pathlib
import subprocess
import tempfile
import xml.etree.ElementTree
from collections.abc import Iterable
from typing import cast, overload

import joblib
import meeko  # type: ignore
import molscrub  # type: ignore
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from ._registry import register


@dataclasses.dataclass(frozen=True)
class AutoDockResult:
    success: bool
    _lowest_binding_energy: float | None = None
    _mean_binding_energy: float | None = None
    _best_pose: str | None = None

    @property
    def lowest_binding_energy(self) -> float:
        if not self.success or self._lowest_binding_energy is None:
            raise ValueError("Docking was not successful, no binding energy available.")
        return self._lowest_binding_energy

    @property
    def mean_binding_energy(self) -> float:
        if not self.success or self._mean_binding_energy is None:
            raise ValueError("Docking was not successful, no binding energy available.")
        return self._mean_binding_energy

    @property
    def best_pose(self) -> str:
        if not self.success or self._best_pose is None:
            raise ValueError("Docking was not successful, no pose available.")
        return self._best_pose


def _parse_result_xml(xml_path: pathlib.Path) -> AutoDockResult:
    if not xml_path.exists():
        return AutoDockResult(success=False)

    tree = xml.etree.ElementTree.ElementTree(xml.etree.ElementTree.fromstring(xml_path.read_text()))
    root = tree.getroot()
    if root is None:
        return AutoDockResult(success=False)

    clusters: list[dict[str, float | int]] = []
    histogram = root.find(".//clustering_histogram")
    if histogram is not None:
        for cluster in histogram.findall("cluster"):
            clusters.append(
                {
                    "cluster_rank": int(cluster.get("cluster_rank", "")),
                    "lowest_binding_energy": float(cluster.get("lowest_binding_energy", "")),
                    "mean_binding_energy": float(cluster.get("mean_binding_energy", "")),
                    "num_in_clus": int(cluster.get("num_in_clus", "")),
                }
            )
    clusters.sort(key=lambda x: x["mean_binding_energy"])

    if not clusters:
        return AutoDockResult(success=False)

    rmsd_table = root.find(".//rmsd_table")
    cluster_to_run_ids: dict[int, list[int]] = {}
    if rmsd_table is None:
        return AutoDockResult(success=False)
    for rmsd in rmsd_table.findall("run"):
        cluster_id = int(rmsd.get("rank", ""))
        run_id = int(rmsd.get("run", ""))
        cluster_to_run_ids.setdefault(cluster_id, []).append(run_id)

    best_run_id = cluster_to_run_ids[1][0]

    all_docked_blocks: list[str] = []
    in_docked: bool = False
    for line in xml_path.with_suffix(".dlg").read_text().splitlines():
        if in_docked:
            if line.startswith("DOCKED: "):
                all_docked_blocks[-1] += line[8:] + "\n"
            else:
                in_docked = False
        else:
            if line.startswith("DOCKED: "):
                all_docked_blocks.append(line[8:] + "\n")
                in_docked = True
            else:
                pass

    return AutoDockResult(
        success=True,
        _lowest_binding_energy=clusters[0]["lowest_binding_energy"],
        _mean_binding_energy=clusters[0]["mean_binding_energy"],
        _best_pose=all_docked_blocks[best_run_id - 1],
    )


def _get_best_result(results: list[AutoDockResult]) -> AutoDockResult:
    if not results:
        return AutoDockResult(success=False)
    return min(results, key=lambda r: r.mean_binding_energy if r.success else float("inf"))


@register
class autodock_gpu:
    def __init__(
        self,
        receptor_path: str | pathlib.Path,
        box_size: tuple[float, float, float],
        box_center: tuple[float, float, float],
        adgpu_path: str | pathlib.Path = "./third_party/bin/adgpu-v1.6_linux_x64_cuda11_128wi",
        extra_args: tuple[str, ...] = ("--nev", "1000000"),
    ) -> None:
        super().__init__()
        self.receptor_path = pathlib.Path(receptor_path).absolute()
        self.box_size = box_size
        self.box_center = box_center
        self.adgpu_path = pathlib.Path(adgpu_path)
        self.extra_args = extra_args

        self.prepare_receptor()

    @property
    def receptor_prefix(self) -> pathlib.Path:
        return self.receptor_path.with_suffix("")

    def prepare_receptor(self) -> None:
        pdbqt_path = self.receptor_path.with_suffix(".pdbqt")
        gpf_path = self.receptor_path.with_suffix(".gpf")
        if not pdbqt_path.exists() or not gpf_path.exists():
            command: list[str] = [
                "mk_prepare_receptor.py",
                "--read_pdb",
                self.receptor_path.name,
                "-o",
                self.receptor_prefix.name,
                "-p",
                "-v",
                "-g",
                "--box_size",
                *map(str, self.box_size),
                "--box_center",
                *map(str, self.box_center),
                "--allow_bad_res",
                "--default_altloc",
                "A",
            ]
            subprocess.run(command, check=True, cwd=self.receptor_path.parent)

            subprocess.run(
                [
                    "autogrid4",
                    "-p",
                    gpf_path.name,
                    "-l",
                    gpf_path.with_suffix(".glg").name,
                ],
                check=True,
                cwd=self.receptor_path.parent,
            )

    def _scrub_ligand(self, mol: Chem.Mol) -> list[Chem.Mol]:
        scrub = molscrub.Scrub()
        try:
            mol_states: list[Chem.Mol] = scrub(mol)
        except RuntimeError:
            mol_states = []
        return mol_states

    def _meeko_ligand(self, mol: Chem.Mol, out_path: pathlib.Path) -> list[pathlib.Path]:
        mk_prep = meeko.MoleculePreparation()
        final_out_paths: list[pathlib.Path] = []
        for i, molsetup in enumerate(mk_prep(mol)):
            final_out_path = out_path.with_suffix(f".{i}.pdbqt")
            final_out_paths.append(final_out_path)
            with final_out_path.open("w") as f:
                f.write(meeko.PDBQTWriterLegacy.write_string(molsetup)[0])
        return final_out_paths

    def prepare_ligand(self, mol: Chem.Mol, ligand_name: str, out_dir: str | pathlib.Path) -> list[pathlib.Path]:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        out_dir = pathlib.Path(out_dir)
        ligand_paths: list[pathlib.Path] = []
        for state_idx, mol_state in enumerate(self._scrub_ligand(mol)):
            ligand_paths.extend(self._meeko_ligand(mol_state, out_path=out_dir / f"{ligand_name}.{state_idx}.pdbqt"))
        return ligand_paths

    def _run_autodock_single(self, ligand_path: pathlib.Path) -> AutoDockResult:
        out = subprocess.run(
            [
                self.adgpu_path.as_posix(),
                "--ffile",
                self.receptor_prefix.with_suffix(".maps.fld").as_posix(),
                "--lfile",
                ligand_path.as_posix(),
                *self.extra_args,
            ],
            check=False,
        )
        xml_path = ligand_path.with_suffix(".xml")
        if out.returncode != 0:
            return AutoDockResult(success=False)
        return _parse_result_xml(xml_path)

    def dock_single(self, mol: Chem.Mol, ligand_name: str, work_dir: str | pathlib.Path) -> AutoDockResult:
        work_dir = pathlib.Path(work_dir).absolute()
        ligand_paths = self.prepare_ligand(mol, ligand_name, work_dir)
        results = [self._run_autodock_single(p) for p in ligand_paths]
        return _get_best_result(results)

    def _run_autodock_multiple(self, ligand_paths: list[pathlib.Path], work_dir: pathlib.Path) -> list[AutoDockResult]:
        maps_path = self.receptor_prefix.with_suffix(".maps.fld")

        batch_lines: list[str] = []
        for ligand_path in ligand_paths:
            batch_lines.extend(["", ligand_path.as_posix(), ligand_path.name])
        batch_lines[0] = maps_path.as_posix()
        batch_file_path = work_dir / "batch.txt"
        batch_file_path.write_text("\n".join(batch_lines))

        subprocess.run(
            [self.adgpu_path.as_posix(), "--filelist", batch_file_path.as_posix(), *self.extra_args],
            check=False,
        )

        results: list[AutoDockResult] = []
        for ligand_path in ligand_paths:
            xml_path = ligand_path.with_suffix(".xml")
            results.append(_parse_result_xml(xml_path))

        return results

    def dock_multiple(self, mols: list[Chem.Mol], work_dir: str | pathlib.Path) -> list[AutoDockResult]:
        work_dir = pathlib.Path(work_dir).absolute()
        nested_paths = cast(
            Iterable[list[pathlib.Path]],
            joblib.Parallel(n_jobs=-1)(
                joblib.delayed(self.prepare_ligand)(mol, f"{i}", work_dir) for i, mol in enumerate(mols)
            ),
        )
        ligand_paths: list[pathlib.Path] = []
        ligand_indices: list[int] = []
        for i, paths in enumerate(nested_paths):
            ligand_paths.extend(paths)
            ligand_indices.extend([i] * len(paths))
        results = self._run_autodock_multiple(ligand_paths, work_dir)

        grouped_results: list[list[AutoDockResult]] = [[] for _ in range(len(mols))]
        for i, result in enumerate(results):
            grouped_results[ligand_indices[i]].append(result)

        best_results = [_get_best_result(g) for g in grouped_results]
        return best_results

    @overload
    def __call__(self, mol: Chem.Mol) -> float: ...
    @overload
    def __call__(self, mol: list[Chem.Mol]) -> list[float]: ...

    def __call__(self, mol: Chem.Mol | list[Chem.Mol]) -> float | list[float]:
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = pathlib.Path(tmpdir)
            if isinstance(mol, Chem.Mol):
                result = self.dock_single(mol, "ligand", work_dir)
                return -1 * result.mean_binding_energy if result.success else 0.0
            else:
                results = self.dock_multiple(mol, work_dir)
                scores = [-1 * r.mean_binding_energy if r.success else 0.0 for r, m in zip(results, mol)]
                return scores


@register
class autodock_Mpro_7gaw(autodock_gpu):
    def __init__(self) -> None:
        super().__init__(
            receptor_path="./data/docking/7gaw/7gaw_atomH.pdb",
            box_size=(20.0, 20.0, 20.0),
            box_center=(-19.0, 5.0, 29.0),
            extra_args=("--nev", "2500000"),
        )
