from pathlib import Path

import click
from tqdm import tqdm

from prexsyn.factory import Config, get_chemical_space, get_data_pipeline, get_detokenizer
from prexsyn.utils.draw import SynthesisDrawer


@click.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", "-o", type=click.Path(file_okay=False, path_type=Path), default=Path("outputs/draw"))
@click.option("--num-samples", "-n", type=int, default=32, help="Number of samples to draw")
def main(config_path: Path, output_dir: Path, num_samples: int):
    config = Config.from_yaml(config_path)
    chemspace = get_chemical_space(config.chemical_space)
    data_pipeline = get_data_pipeline(config, chemspace=chemspace)
    data_pipeline.start_workers([2026])
    batch = data_pipeline.get(num_samples)

    detokenizer = get_detokenizer(config, chemspace=chemspace)
    detok = detokenizer(batch["synthesis"])

    output_dir.mkdir(parents=True, exist_ok=True)
    with SynthesisDrawer() as drawer:
        for i, syn in enumerate(tqdm(detok, desc="Drawing syntheses")):
            img = drawer.draw(syn)
            img.save(output_dir / f"synthesis_{i}.png")


if __name__ == "__main__":
    main()
