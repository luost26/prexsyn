from pathlib import Path

from prexsyn.utils.download import download

from .config import Config


def download_chemical_space_if_needed(config: Config):
    cs_conf = config.chemical_space
    remote_conf = config.remote
    if not cs_conf.cache_path.exists() and remote_conf.chemical_space_url is not None:
        download(remote_conf.chemical_space_url, cs_conf.cache_path, desc="Downloading chemical space")


def download_checkpoint_if_needed(config: Config, expected_path: Path):
    remote_conf = config.remote
    if not expected_path.exists() and remote_conf.checkpoint_url is not None:
        download(remote_conf.checkpoint_url, expected_path, desc="Downloading checkpoint")
