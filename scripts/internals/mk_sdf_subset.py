import click
from rdkit import Chem


@click.command()
@click.option("-i", "--input_sdf", type=click.Path(exists=True), required=True)
@click.option("-o", "--output_sdf", type=click.Path(), required=True)
@click.option("-n", "--num_molecules", type=int, default=200)
def main(input_sdf, output_sdf, num_molecules):
    suppl = Chem.SDMolSupplier(input_sdf)

    molecules = []
    for mol in suppl:
        if mol is None:
            continue
        molecules.append(mol)
        if len(molecules) >= num_molecules:
            break

    writer = Chem.SDWriter(output_sdf)
    for mol in molecules:
        writer.write(mol)
    writer.close()


if __name__ == "__main__":
    main()
