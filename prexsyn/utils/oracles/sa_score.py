# flake8: noqa
# type: ignore
# fmt: off
#
# calculation of synthetic accessibility score as described in:
#
# Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
# Peter Ertl and Ansgar Schuffenhauer
# Journal of Cheminformatics 1:8 (2009)
# http://www.jcheminf.com/content/1/1/8
#
# several small modifications to the original paper are included
# particularly slightly different formula for marocyclic penalty
# and taking into account also molecule symmetry (fingerprint density)
#
# for a set of 10k diverse molecules the agreement between the original method
# as implemented in PipelinePilot and this implementation is r2 = 0.97
#
# peter ertl & greg landrum, september 2013
#

import math
import os.path as op
import pickle

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

_fscores = None
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)


def readFragmentScores(name="fpscores.pkl.gz"):
  import gzip
  global _fscores
  # generate the full path filename:
  if name == "fpscores.pkl.gz":
    name = op.join(op.dirname(__file__), name)
  data = pickle.load(gzip.open(name))
  outDict = {}
  for i in data:
    for j in range(1, len(i)):
      outDict[i[j]] = float(i[0])
  _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
  nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
  nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
  return nBridgehead, nSpiro


def calculateScore(m, fname="fpscores.pkl.gz"):

  if not m.GetNumAtoms():
    return None

  if _fscores is None:
    readFragmentScores(fname)

  # fragment score
  sfp = mfpgen.GetSparseCountFingerprint(m)

  score1 = 0.
  nf = 0
  nze = sfp.GetNonzeroElements()
  for id, count in nze.items():
    nf += count
    score1 += _fscores.get(id, -4) * count

  score1 /= nf

  # features score
  nAtoms = m.GetNumAtoms()
  nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
  ri = m.GetRingInfo()
  nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
  nMacrocycles = 0
  for x in ri.AtomRings():
    if len(x) > 8:
      nMacrocycles += 1

  sizePenalty = nAtoms**1.005 - nAtoms
  stereoPenalty = math.log10(nChiralCenters + 1)
  spiroPenalty = math.log10(nSpiro + 1)
  bridgePenalty = math.log10(nBridgeheads + 1)
  macrocyclePenalty = 0.
  # ---------------------------------------
  # This differs from the paper, which defines:
  #  macrocyclePenalty = math.log10(nMacrocycles+1)
  # This form generates better results when 2 or more macrocycles are present
  if nMacrocycles > 0:
    macrocyclePenalty = math.log10(2)

  score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

  # correction for the fingerprint density
  # not in the original publication, added in version 1.1
  # to make highly symmetrical molecules easier to synthetise
  score3 = 0.
  numBits = len(nze)
  if nAtoms > numBits:
    score3 = math.log(float(nAtoms) / numBits) * .5

  sascore = score1 + score2 + score3

  # need to transform "raw" value into scale between 1 and 10
  min = -4.0
  max = 2.5
  sascore = 11. - (sascore - min + 1) / (max - min) * 9.

  # smooth the 10-end
  if sascore > 8.:
    sascore = 8. + math.log(sascore + 1. - 9.)
  if sascore > 10.:
    sascore = 10.0
  elif sascore < 1.:
    sascore = 1.0

  return sascore


def processMols(mols):
  print('smiles\tName\tsa_score')
  for i, m in enumerate(mols):
    if m is None:
      continue

    s = calculateScore(m)

    smiles = Chem.MolToSmiles(m)
    if s is None:
      print(f"{smiles}\t{m.GetProp('_Name')}\t{s}")
    else:
      print(f"{smiles}\t{m.GetProp('_Name')}\t{s:3f}")

#
#  Copyright (c) 2013, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc.
#       nor the names of its contributors may be used to endorse or promote
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# fmt: on

import pathlib

import requests
from tqdm import tqdm

from ._registry import register


def download(local: str | pathlib.Path) -> None:
    remote = "https://raw.githubusercontent.com/rdkit/rdkit/refs/heads/master/Contrib/SA_Score/fpscores.pkl.gz"
    response = requests.get(remote)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    print("Downloading SA score model...")
    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
        with open(local, "wb") as file:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                file.write(data)
    print("Download complete.")

    if total_size != 0 and pbar.n != total_size:
        raise RuntimeError(f"Failed to download file from: {remote}")


@register
class sa_score:
    def __init__(self, normalized: bool = True) -> None:
        super().__init__()
        self.normalized = normalized
        model_path = pathlib.Path("./data/oracles/sa_score.pkl.gz")
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            download(model_path)
        self.model_path = model_path

    def _score(self, mol: Chem.Mol) -> float:
        score = calculateScore(mol, self.model_path.as_posix())
        if self.normalized:
            score = (10.0 - score) / 9.0  # normalize to [0,1], 1 is easy to make
        return score

    def __call__(self, mol: list[Chem.Mol] | Chem.Mol) -> list[float] | float:
        single = not isinstance(mol, list)
        if not isinstance(mol, list):
            mol = [mol]
        results = [self._score(m) for m in mol]
        return results[0] if single else results
