from collections.abc import Mapping
from typing import Any

from prexsyn_engine.featurizer.synthesis import PostfixNotationTokenDef


class Tokenization:
    def __init__(self, token_def: PostfixNotationTokenDef = PostfixNotationTokenDef()) -> None:
        self._token_def = token_def

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "Tokenization":
        token_def = PostfixNotationTokenDef(**cfg.get("def", {}))
        return cls(token_def=token_def)

    @property
    def token_def(self) -> PostfixNotationTokenDef:
        return self._token_def
