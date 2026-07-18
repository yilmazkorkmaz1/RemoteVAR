"""Protect Accelerate's Hugging Face ``datasets`` import from local name collisions."""

from importlib import metadata, util
from pathlib import Path
import sys


def _module_is_from_directory(module, directory: Path) -> bool:
    """Return whether ``module`` was loaded from ``directory``."""
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        return Path(module_file).resolve().parent == directory
    except OSError:
        return False


def ensure_huggingface_datasets() -> None:
    """
    Load the installed Hugging Face ``datasets`` package into ``sys.modules``.

    RemoteVAR and its parent repository both contain project-local packages named
    ``datasets``. Accelerate imports ``datasets.IterableDataset`` while preparing
    a distributed DataLoader, so normal Python path precedence can otherwise
    resolve one of those local packages instead of Hugging Face datasets.
    """
    try:
        package_dir = Path(metadata.distribution("datasets").locate_file("datasets")).resolve()
    except metadata.PackageNotFoundError:
        # Accelerate handles the optional dependency being absent.
        return

    init_file = package_dir / "__init__.py"
    if not init_file.is_file():
        raise ImportError(f"Hugging Face datasets package is incomplete: {init_file} is missing.")

    loaded_module = sys.modules.get("datasets")
    if loaded_module is not None and _module_is_from_directory(loaded_module, package_dir):
        return

    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "datasets" or name.startswith("datasets.")
    }
    for name in previous_modules:
        del sys.modules[name]

    spec = util.spec_from_file_location(
        "datasets",
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        sys.modules.update(previous_modules)
        raise ImportError(f"Could not load Hugging Face datasets from {init_file}.")

    module = util.module_from_spec(spec)
    sys.modules["datasets"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        for name in tuple(sys.modules):
            if name == "datasets" or name.startswith("datasets."):
                del sys.modules[name]
        sys.modules.update(previous_modules)
        raise
