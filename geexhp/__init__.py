from importlib import import_module

__version__ = "1.0.0"

_MODULE_MAP = {
    "datavis": "geexhp.core.datavis",
    "datagen": "geexhp.core.datagen",
    "datamod": "geexhp.core.datamod",
    "stages": "geexhp.core.stages",
    "datasetup": "geexhp.modelfuncs.datasetup",
    "sabcnn_DEPRECATED": "geexhp.modelfuncs.sabcnn_DEPRECATED",
    "tfrecord_conversion": "geexhp.modelfuncs.tfrecord_conversion",
}

__all__ = list(_MODULE_MAP)


def __getattr__(name: str):
    if name in _MODULE_MAP:
        module = import_module(_MODULE_MAP[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'geexhp' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + list(_MODULE_MAP.keys()))
