from pathlib import Path

import click
from tqdm.auto import tqdm

from prexsyn.factory import Config
from prexsyn.training.data_module import SynthesisDataStream


@click.command()
@click.argument("config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--inspect", is_flag=True)
def main(config_path: Path, inspect: bool):
    config = Config.from_yaml(config_path)
    print(config)

    data_stream = SynthesisDataStream(config, 1000, list(range(16)))
    for batch in tqdm(data_stream):
        if inspect:
            print(batch)
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
