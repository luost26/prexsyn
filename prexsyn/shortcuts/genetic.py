import io
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import PIL.Image
import pydot

from prexsyn.utils.draw import draw_fingerprint, draw_molecule
from prexsyn_engine.chemistry import Molecule
from prexsyn_engine.chemspace import Synthesis
from prexsyn_engine.descriptor import _MoleculeDescriptor

from .projector import MoleculeProjector


@dataclass
class Population:
    genotypes: np.ndarray
    phenotypes: list[tuple[Synthesis, Molecule]]
    fitnesses: np.ndarray
    unique_identifiers: np.ndarray
    parents: np.ndarray

    def size(self):
        return len(self.genotypes)

    def concat(self, other: "Population") -> "Population":
        return Population(
            genotypes=np.concatenate([self.genotypes, other.genotypes], axis=0),
            phenotypes=self.phenotypes + other.phenotypes,
            fitnesses=np.concatenate([self.fitnesses, other.fitnesses], axis=0),
            unique_identifiers=np.concatenate([self.unique_identifiers, other.unique_identifiers], axis=0),
            parents=np.concatenate([self.parents, other.parents], axis=0),
        )

    def index_select(self, indices: np.ndarray | list[int]) -> "Population":
        return Population(
            genotypes=self.genotypes[indices],
            phenotypes=[self.phenotypes[i] for i in indices],
            fitnesses=self.fitnesses[indices],
            unique_identifiers=self.unique_identifiers[indices],
            parents=self.parents[indices],
        )

    def dedup(self):
        seen = set()
        unique_indices = []
        for i, (_, mol) in enumerate(self.phenotypes):
            if mol.smiles not in seen:
                seen.add(mol.smiles)
                unique_indices.append(i)
        return self.index_select(unique_indices)

    def topk(self, k: int):
        topk_indices = np.argsort(self.fitnesses)[-k:]
        return self.index_select(topk_indices)

    def assign(self, other: "Population"):
        self.genotypes = other.genotypes
        self.phenotypes = other.phenotypes
        self.fitnesses = other.fitnesses
        self.unique_identifiers = other.unique_identifiers
        self.parents = other.parents


@dataclass
class EmbryoSet:
    genotypes: np.ndarray
    unique_identifiers: np.ndarray
    parents: np.ndarray


def _softmax(x: np.ndarray, t: float):
    x = x / t
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def select_and_recombine_genotypes(ppl: Population, k: int, t: float, uniq_start: int):
    prob = _softmax(ppl.fitnesses, t)
    selected_indices = np.random.choice(len(ppl.genotypes), size=k, replace=False, p=prob)
    parent_1_indices = np.random.choice(selected_indices, size=k, replace=True)
    parent_2_indices = np.random.choice(selected_indices, size=k, replace=True)

    parent_1_uniques, parent_1_genotypes = (
        ppl.unique_identifiers[parent_1_indices],
        ppl.genotypes[parent_1_indices],
    )
    parent_2_uniques, parent_2_genotypes = (
        ppl.unique_identifiers[parent_2_indices],
        ppl.genotypes[parent_2_indices],
    )

    if ppl.genotypes.dtype == bool:
        bit_prob = 0.5 * (parent_1_genotypes.astype(float) + parent_2_genotypes.astype(float))
        new_genotypes = np.random.rand(*parent_1_genotypes.shape) < bit_prob
        parent_uniques = np.stack([parent_1_uniques, parent_2_uniques], axis=1)
    else:
        raise NotImplementedError("Only boolean genotypes are supported for now.")

    new_uniques = np.arange(uniq_start, uniq_start + len(new_genotypes))

    return EmbryoSet(genotypes=new_genotypes, unique_identifiers=new_uniques, parents=parent_uniques)


_FitnessFunction: TypeAlias = Callable[[Sequence[tuple[Synthesis, Molecule]]], np.ndarray]


def hatch(
    em: EmbryoSet,
    projector: MoleculeProjector,
    reverse_fn: _MoleculeDescriptor,
    fitness_fn: _FitnessFunction,
):
    results = projector.desc(em.genotypes)
    phenotypes: list[tuple[Synthesis, Molecule]] = []

    hatched_indices: list[int] = []
    for i in range(len(results)):
        result = results[i]
        best = result.best()
        if best is not None:
            phenotypes.append((best.synthesis, best.molecule))
            hatched_indices.append(i)

    mols = [mol for _, mol in phenotypes]
    genotypes = reverse_fn(mols)
    fitnesses = fitness_fn(phenotypes)

    parents = em.parents[hatched_indices]
    unique_ids = em.unique_identifiers[hatched_indices]

    return Population(
        genotypes=genotypes,
        phenotypes=phenotypes,
        fitnesses=fitnesses,
        unique_identifiers=unique_ids,
        parents=parents,
    )


@dataclass
class Individual:
    genotype: np.ndarray
    phenotype: tuple[Synthesis, Molecule]
    fitness: float
    unique_id: int
    parent_ids: tuple[int, ...]


class History:
    def __init__(self):
        self.individuals: dict[int, Individual] = {}
        self.max_unique_id = -1

    def add_population(self, population: Population):
        for i in range(population.size()):
            individual = Individual(
                genotype=population.genotypes[i],
                phenotype=population.phenotypes[i],
                fitness=float(population.fitnesses[i]),
                unique_id=int(population.unique_identifiers[i]),
                parent_ids=tuple(int(p) for p in population.parents[i]),
            )
            self.individuals[individual.unique_id] = individual
            self.max_unique_id = max(self.max_unique_id, individual.unique_id)

    def next_unique_id(self):
        return self.max_unique_id + 1


class EvolutionaryTreeDraw:
    """Draw evolutionary trees for individuals in genetic algorithm history."""

    def __init__(self):
        self.rankdir = "TB"
        self.fontname = "Fira Sans"
        self.dpi = 200
        self.bgcolor = "transparent"

    def _get_individual_by_id(self, history: History, unique_id: int) -> Individual | None:
        """Find an individual in history by unique_id."""
        return history.individuals.get(unique_id, None)

    def _collect_tree_nodes(
        self, individual: Individual, history: History, visited: set[int] | None = None
    ) -> set[int]:
        """Recursively collect all node IDs in the evolutionary tree rooted at individual."""
        if visited is None:
            visited = set()

        if individual.unique_id in visited:
            return visited

        visited.add(individual.unique_id)

        for parent_id in individual.parent_ids:
            if parent_id == -1:
                continue
            parent = self._get_individual_by_id(history, parent_id)
            if parent is not None:
                self._collect_tree_nodes(parent, history, visited)

        return visited

    def _individual_node(
        self,
        node_id: str,
        individual: Individual,
        working_dir: Path,
    ) -> pydot.Node:
        """Create a pydot node representing an individual with molecule and fingerprint."""
        mol_path = working_dir / f"{node_id}_mol.png"
        fp_path = working_dir / f"{node_id}_fp.png"

        # Draw molecule and fingerprint
        _, mol = individual.phenotype
        draw_molecule(mol).save(mol_path)
        draw_fingerprint(individual.genotype, dpi=72).save(fp_path)

        label_lines = ["<"]
        label_lines += [
            '<TABLE STYLE="ROUNDED" BORDER="0" CELLBORDER="0" CELLSPACING="5" CELLPADDING="0" BGCOLOR="grey97">',
            '<TR><TD><IMG SRC="' + mol_path.as_posix() + '"/></TD></TR>',
            '<TR><TD><IMG SRC="' + fp_path.as_posix() + '"/></TD></TR>',
            f"<TR><TD>ID: {individual.unique_id}</TD></TR>",
            f"<TR><TD>Fitness: {individual.fitness:.4f}</TD></TR>",
            "</TABLE>",
            ">",
        ]

        return pydot.Node(
            node_id,
            shape="plaintext",
            label="".join(label_lines),
            fontsize="10",
            fontname=self.fontname,
        )

    def draw(
        self,
        individual: Individual,
        history: History,
        pdf_output: Path | None = None,
    ) -> PIL.Image.Image:
        """Draw the evolutionary tree for a specific individual.

        Args:
            individual: The individual whose ancestry to visualize
            history: The evolutionary history containing all individuals
            pdf_output: Optional path to save PDF output

        Returns:
            PIL.Image.Image: The rendered tree visualization
        """
        node_id_to_str: dict[int, str] = {}

        def _get_node_id(unique_id: int) -> str:
            if unique_id not in node_id_to_str:
                node_id_str = f"ind_{len(node_id_to_str)}"
                node_id_to_str[unique_id] = node_id_str
            return node_id_to_str[unique_id]

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
                nodesep=0.1,
            )

            # Collect all nodes in the tree
            tree_node_ids = self._collect_tree_nodes(individual, history)

            # Add all nodes to the graph
            for unique_id in tree_node_ids:
                ind = self._get_individual_by_id(history, unique_id)
                if ind is not None:
                    P.add_node(self._individual_node(_get_node_id(unique_id), ind, working_dir))

            # Add edges from parents to children
            for unique_id in tree_node_ids:
                ind = self._get_individual_by_id(history, unique_id)
                if ind is not None:
                    for parent_id in ind.parent_ids:
                        if parent_id == -1:
                            continue
                        if parent_id in tree_node_ids:
                            P.add_edge(
                                pydot.Edge(
                                    P.get_node(_get_node_id(parent_id))[0],
                                    P.get_node(_get_node_id(unique_id))[0],
                                    fontsize=8,
                                    fontname=self.fontname,
                                )
                            )

            if pdf_output is not None:
                P.write_pdf(pdf_output)  # type: ignore[attr-defined]

            return PIL.Image.open(io.BytesIO(P.create_png()))  # type: ignore[attr-defined]


def initialize(size: int, projector: MoleculeProjector, fn: _FitnessFunction):
    if projector.descriptor_function.dtype() != np.dtype(bool):
        raise NotImplementedError("Only boolean genotypes are supported for now.")
    rand_genotypes = np.random.rand(size * 2, *projector.descriptor_function.size()) < 0.02

    embryos = EmbryoSet(
        genotypes=rand_genotypes,
        unique_identifiers=np.arange(size * 2),
        parents=np.full((size * 2, 2), -1),
    )

    population = hatch(embryos, projector, projector.descriptor_function, fn)
    history = History()
    history.add_population(population)
    return population, history


def evolve(
    ppl: Population,
    history: History,
    projector: MoleculeProjector,
    fitness_fn: _FitnessFunction,
    k: int,
    t: float,
) -> None:
    em = select_and_recombine_genotypes(ppl, k, t, uniq_start=history.next_unique_id())
    new_ppl = hatch(em, projector, projector.descriptor_function, fitness_fn)
    history.add_population(new_ppl)
    ppl.assign(ppl.concat(new_ppl).dedup().topk(k))
