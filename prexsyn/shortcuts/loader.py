import functools
from pathlib import Path

from prexsyn.factory import Config, get_chemical_space, get_detokenizer, load_model
from prexsyn.utils.download import download


class AllInOneLoader:
    def __init__(self, config_path: str | Path):
        super().__init__()
        path_str = str(config_path)
        is_remote = path_str.startswith("http://") or path_str.startswith("https://")
        if is_remote:
            base_name = Path(path_str).name
            local_config_path = Path("./data/trained_models") / base_name
            local_config_path.parent.mkdir(parents=True, exist_ok=True)
            download(path_str, config_path, desc="Downloading config")
            config_path = local_config_path
        self._config_path = Path(config_path)
        self._config = Config.from_yaml(self._config_path)

    def config(self):
        return self._config

    def config_path(self):
        return self._config_path

    @functools.cache
    def model(self):
        ckpt_path = self._config_path.with_suffix(".ckpt")

        remote_conf = self._config.remote
        if not ckpt_path.exists():
            if remote_conf.checkpoint_url is not None:
                download(remote_conf.checkpoint_url, ckpt_path, desc="Downloading checkpoint")
            else:
                raise FileNotFoundError(f"Checkpoint not found at {ckpt_path} and no remote URL provided.")

        return load_model(self._config, ckpt_path).eval()

    @functools.cache
    def chemical_space(self):
        cs_path = self._config.chemical_space.cache_path

        if not cs_path.exists():
            cs_url = self._config.remote.chemical_space_url
            if cs_url is not None:
                download(cs_url, cs_path, desc="Downloading chemical space")
            else:
                raise FileNotFoundError(f"Chemical space not found at {cs_path} and no remote URL provided.")

        return get_chemical_space(self._config.chemical_space)

    @functools.cache
    def detokenizer(self, **kwargs):
        cs = self.chemical_space()
        return get_detokenizer(self._config, chemspace=cs, **kwargs)
