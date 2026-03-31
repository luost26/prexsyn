from collections.abc import Callable

import prexsyn_engine

_descriptor_registry: dict[str, Callable[[], prexsyn_engine.descriptor._MoleculeDescriptor]] = {
    "ecfp4": prexsyn_engine.descriptor.MorganFingerprint.ecfp4,
    "fcfp4": prexsyn_engine.descriptor.MorganFingerprint.fcfp4,
}


def get_descriptor_constructor(t: str):
    return _descriptor_registry[t]


def register_descriptor(t: str, constructor: Callable[[], prexsyn_engine.descriptor._MoleculeDescriptor]):
    if t in _descriptor_registry:
        raise ValueError(f"Descriptor '{t}' is already registered.")
    _descriptor_registry[t] = constructor
