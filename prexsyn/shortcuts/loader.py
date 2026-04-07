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
            local_config_path = Path("./data/trained_models/remote") / base_name
            local_config_path.parent.mkdir(parents=True, exist_ok=True)
            download(path_str, local_config_path, desc="Downloading config")
            config_path = local_config_path
        self._config_path = Path(config_path)
        self._config = Config.from_yaml(self._config_path)
        self._ckpt_path = self._config_path.with_suffix(".ckpt")
        self._cs_path = self._config.chemical_space.cache_path

        self.print_info()

    def print_info(self):
        print("[PrexSyn All-in-One Loader]")
        print(f"- Model name: {self._config.note.name}")
        print(f"- Config path: {self._config_path}")
        print(f"- Checkpoint path: {self._ckpt_path}")
        print(f"- Chemspace path: {self._cs_path}")
        print("- Description:")
        for line in self._config.note.description.splitlines():
            print(f"  > {line}")
        print("")

    def config(self):
        return self._config

    def config_path(self):
        return self._config_path

    def ensure_chemical_space(self):
        cs_path = self._cs_path
        if not cs_path.exists():
            cs_url = self._config.remote.chemical_space_url
            if cs_url is not None:
                download(cs_url, cs_path, desc="Downloading chemical space")
            else:
                raise FileNotFoundError(f"Chemical space not found at {cs_path} and no remote URL provided.")

    @functools.cache
    def model(self):
        # The model peeks into chemical space to determine embedding layer size
        self.ensure_chemical_space()
        ckpt_path = self._ckpt_path

        remote_conf = self._config.remote
        if not ckpt_path.exists():
            if remote_conf.checkpoint_url is not None:
                download(remote_conf.checkpoint_url, ckpt_path, desc="Downloading checkpoint")
            else:
                raise FileNotFoundError(f"Checkpoint not found at {ckpt_path} and no remote URL provided.")

        return load_model(self._config, ckpt_path).eval()

    @functools.cache
    def chemical_space(self):
        self.ensure_chemical_space()
        return get_chemical_space(self._config.chemical_space)

    @functools.cache
    def detokenizer(self, **kwargs):
        cs = self.chemical_space()
        return get_detokenizer(self._config, chemspace=cs, **kwargs)
