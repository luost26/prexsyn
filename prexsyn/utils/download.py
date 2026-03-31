import pathlib

import requests
from tqdm import tqdm


def download(remote: str, local: str | pathlib.Path, desc: str = "Downloading") -> None:
    response = requests.get(remote)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
        with open(local, "wb") as file:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                file.write(data)

    if total_size != 0 and pbar.n != total_size:
        raise RuntimeError(f"Failed to download file from: {remote}")
