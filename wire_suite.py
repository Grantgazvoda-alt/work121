"""Load the installed work122 ``agent-comm-robustness`` distribution.

Evaluation code imports the wire suite through this module. The in-repo
``agent_comm_robustness/`` directory is retained for a later removal and is
not the eval transport. To move to another work122 revision, change the single
git pin in ``pyproject.toml`` and reinstall; callers keep importing
``wire_suite.load_suite``.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

DISTRIBUTION_NAME = "agent-comm-robustness"
GIT_PIN = "72960f25d8acd3c3152c32b139425f4d3d806d85"
_ALIAS = "work122_agent_comm_robustness"
_REQUIRED_EXPORTS = (
    "AgentMessage",
    "FramingMode",
    "HardenedBoundaryGate",
    "MessageEncoder",
    "PayloadCategory",
    "PayloadVariant",
    "Performative",
    "Provenance",
    "sign_observed_claim",
)


def install_hint() -> str:
    """Command that installs the pinned work122 distribution."""
    return (
        "pip install "
        f"'{DISTRIBUTION_NAME} @ git+https://github.com/Grantgazvoda-alt/work122.git@{GIT_PIN}'"
    )


def load_suite() -> ModuleType:
    """Return the installed work122 package, cached under a private module name."""
    cached = sys.modules.get(_ALIAS)
    if cached is not None and all(hasattr(cached, name) for name in _REQUIRED_EXPORTS):
        return cached

    try:
        dist = importlib.metadata.distribution(DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError(
            f"The {DISTRIBUTION_NAME} distribution from work122 is not installed. {install_hint()}"
        ) from exc

    init_path = _distribution_init(dist)
    repo_init = (Path(__file__).resolve().parent / "agent_comm_robustness" / "__init__.py").resolve()
    if init_path.resolve() == repo_init:
        raise ImportError(
            "agent-comm-robustness resolved to work121's retained agent_comm_robustness/ copy. "
            f"Install work122 instead. {install_hint()}"
        )

    spec = importlib.util.spec_from_file_location(
        _ALIAS,
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the work122 suite from {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_ALIAS] = module
    spec.loader.exec_module(module)
    missing = [name for name in _REQUIRED_EXPORTS if not hasattr(module, name)]
    if missing:
        raise ImportError(
            f"{DISTRIBUTION_NAME} at {init_path} is missing exports: {', '.join(missing)}"
        )
    return module


def suite_version() -> str:
    """Installed distribution version of the work122 wire suite."""
    load_suite()
    return importlib.metadata.version(DISTRIBUTION_NAME)


def suite_origin() -> Path:
    """Filesystem path of the loaded suite's ``__init__.py``."""
    module = load_suite()
    file_name = getattr(module, "__file__", None)
    if not file_name:
        raise ImportError("Loaded work122 suite has no __file__")
    return Path(file_name).resolve()


def _distribution_init(dist: importlib.metadata.Distribution) -> Path:
    for entry in dist.files or []:
        if entry.as_posix().endswith("agent_comm_robustness/__init__.py"):
            return Path(str(dist.locate_file(entry)))
    raise ImportError(
        f"{DISTRIBUTION_NAME} {dist.version} does not contain agent_comm_robustness/__init__.py"
    )
