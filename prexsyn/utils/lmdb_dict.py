from __future__ import annotations

import pickle
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Callable, Generic, TypeVar, cast

import lmdb

K = TypeVar("K")
V = TypeVar("V")


def _pickle_dumps(obj: object) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def _pickle_loads(data: bytes) -> object:
    return pickle.loads(data)


class LmdbDict(MutableMapping[K, V], Generic[K, V]):
    """A persistent dict-like object backed by LMDB.

    By default, keys and values are serialized with pickle, allowing arbitrary
    Python objects to be used as mapping entries.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        map_size: int = 1 << 40,  # 1 TB
        readonly: bool = False,
        create: bool = True,
        key_encoder: Callable[[K], bytes] | None = None,
        key_decoder: Callable[[bytes], K] | None = None,
        value_encoder: Callable[[V], bytes] | None = None,
        value_decoder: Callable[[bytes], V] | None = None,
        sync: bool = True,
    ) -> None:
        self.path = Path(path)
        self.readonly = readonly

        if readonly and not self.path.exists():
            raise FileNotFoundError(f"LMDB path does not exist: {self.path}")

        if not readonly and create:
            self.path.parent.mkdir(parents=True, exist_ok=True)

        self._key_encoder: Callable[[K], bytes] = key_encoder or cast(Callable[[K], bytes], _pickle_dumps)
        self._key_decoder: Callable[[bytes], K] = key_decoder or cast(Callable[[bytes], K], _pickle_loads)
        self._value_encoder: Callable[[V], bytes] = value_encoder or cast(Callable[[V], bytes], _pickle_dumps)
        self._value_decoder: Callable[[bytes], V] = value_decoder or cast(Callable[[bytes], V], _pickle_loads)

        self._env = lmdb.open(
            str(self.path),
            map_size=map_size,
            subdir=False,
            readonly=readonly,
            create=create,
            lock=not readonly,
            readahead=readonly,
            writemap=False,
            metasync=sync,
            sync=sync,
            meminit=False,
        )

    def _ensure_writable(self) -> None:
        if self.readonly:
            raise TypeError("Cannot mutate a read-only LmdbDict")

    def close(self) -> None:
        self._env.close()

    def sync(self) -> None:
        self._env.sync()

    def __enter__(self) -> LmdbDict[K, V]:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __getitem__(self, key: K) -> V:
        raw_key = self._key_encoder(key)
        with self._env.begin(write=False) as txn:
            raw_value = txn.get(raw_key)

        if raw_value is None:
            raise KeyError(key)

        return self._value_decoder(raw_value)

    def __setitem__(self, key: K, value: V) -> None:
        self._ensure_writable()
        raw_key = self._key_encoder(key)
        raw_value = self._value_encoder(value)
        with self._env.begin(write=True) as txn:
            txn.put(raw_key, raw_value)

    def __delitem__(self, key: K) -> None:
        self._ensure_writable()
        raw_key = self._key_encoder(key)
        with self._env.begin(write=True) as txn:
            deleted = txn.delete(raw_key)

        if not deleted:
            raise KeyError(key)

    def __iter__(self) -> Iterator[K]:
        with self._env.begin(write=False) as txn:
            cursor = txn.cursor()
            for raw_key, _ in cursor:
                yield self._key_decoder(raw_key)

    def __len__(self) -> int:
        with self._env.begin(write=False) as txn:
            return txn.stat()["entries"]

    def clear(self) -> None:
        self._ensure_writable()
        with self._env.begin(write=True) as txn:
            txn.drop(db=None, delete=False)
