import importlib
import inspect
import logging
from pathlib import Path
from brain.adapters.base import BaseAdapter

logger = logging.getLogger(__name__)


def load_adapters() -> dict:
    here = Path(__file__).parent
    registry = {}
    for plugin_dir in sorted(here.iterdir()):
        if not plugin_dir.is_dir() or not (plugin_dir / "adapter.py").exists():
            continue
        module_path = f"brain.adapters.{plugin_dir.name}.adapter"
        try:
            mod = importlib.import_module(module_path)
        except Exception as e:
            logger.warning("Failed to import '%s': %s", module_path, e)
            continue
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if (
                issubclass(cls, BaseAdapter)
                and cls is not BaseAdapter
                and getattr(cls, "ENABLED", True)
            ):
                registry[cls.PROVIDER_KEY] = cls()

    logger.info("Loaded %d adapters: %s", len(registry), sorted(registry.keys()))
    return registry


REGISTRY = load_adapters()
ALL_ADAPTERS = [type(v) for v in REGISTRY.values()]
__all__ = ["REGISTRY", "ALL_ADAPTERS"]
