import pathlib

from prexsyn_engine.building_block_list import BuildingBlockList
from prexsyn_engine.chemspace import (
    ChemicalSpaceDefinition,
    ChemicalSpaceDefinitionBuilder,
)
from prexsyn_engine.reaction_list import ReactionList


class ChemicalSpace:
    _data_files = (
        "primary_building_blocks",
        "primary_index",
        "reactions",
        "secondary_building_blocks",
        "secondary_index",
    )

    def __init__(self, data_dir: str | pathlib.Path) -> None:
        super().__init__()
        self._data_dir = pathlib.Path(data_dir)

        for file in self._data_files:
            if not (self._data_dir / file).exists():
                raise FileNotFoundError(
                    f"Required data file '{file}' not found in '{data_dir}'. "
                    "Please remove the directory and re-generate the chemical space cache."
                )

        self._csd: ChemicalSpaceDefinition | None = None

    @classmethod
    def load_or_create(
        cls,
        building_block_path: str | pathlib.Path,
        reaction_path: str | pathlib.Path,
        cache_dir: str | pathlib.Path,
    ) -> "ChemicalSpace":
        building_block_path = pathlib.Path(building_block_path)
        reaction_path = pathlib.Path(reaction_path)
        cache_dir = pathlib.Path(cache_dir)
        exists = list((cache_dir / file_name).exists() for file_name in cls._data_files)

        if not any(exists):
            csd = (
                ChemicalSpaceDefinitionBuilder()
                .building_blocks_from_sdf(building_block_path.as_posix())
                .reactions_from_txt(reaction_path.as_posix())
                .secondary_building_blocks_from_single_reaction()
                .build_primary_index()
                .build_secondary_index()
                .build()
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            csd.save(str(cache_dir))
        elif not all(exists):
            missing = [file_name for file_name, file_exists in zip(cls._data_files, exists) if not file_exists]
            raise FileNotFoundError(
                f"Cache directory '{cache_dir}' is incomplete. Missing files: {', '.join(missing)}. "
                "Please remove the directory and re-generate the chemical space cache."
            )

        return cls(cache_dir)

    def count_building_blocks(self) -> int:
        if self._csd is None:
            return BuildingBlockList.peek_size((self._data_dir / "primary_building_blocks").as_posix())
        return len(self._csd.get_primary_building_blocks())

    def count_reactions(self) -> int:
        if self._csd is None:
            return ReactionList.peek_size((self._data_dir / "reactions").as_posix())
        return len(self._csd.get_reactions())

    def get_csd(self) -> ChemicalSpaceDefinition:
        if self._csd is None:
            self._csd = ChemicalSpaceDefinitionBuilder().all_from_cache((self._data_dir).as_posix()).build()
        return self._csd
