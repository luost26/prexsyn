import rdkit.Chem.rdchem
import pandas as pd

from prexsyn_engine.fingerprints import fp_func

if __name__ == "__main__":
    df = pd.read_csv("data/benchmarks/chembl_1k.txt")

    all_fp_types = ["ecfp4", "fcfp4", "rdkit"]
    ref_results = {"smiles": []}
    for fp_type in all_fp_types:
        ref_results[fp_type] = []

    for smi in df["SMILES"][:100]:
        mol = rdkit.Chem.MolFromSmiles(smi)
        ref_results["smiles"].append(smi)
        for fp_type in all_fp_types:
            fp = fp_func(mol, fp_type)
            nz = fp.nonzero()[0].tolist()
            ref_results[fp_type].append(";".join(map(str, nz)))
    ref_df = pd.DataFrame(ref_results)
    print(ref_df)

    ref_df.to_csv("data/benchmarks/fp_ref.csv", index=False)
