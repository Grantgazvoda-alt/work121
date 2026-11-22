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
_DEFAULT_GIT_PIN = "86b3509eee32a1ddb85ac36eecd1bb913952fa92"
_PIN_PREFIX = f"{DISTRIBUTION_NAME} @ git+https://github.com/Grantgazvoda-alt/work122.git@"


def _read_git_pin() -> str:
    pyproject = Path(__file__).resolve().with_name("pyproject.toml")
    if pyproject.is_file():
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            if _PIN_PREFIX in line:
                pin = line.split(_PIN_PREFIX, 1)[1].split('"', 1)[0].split("'", 1)[0].strip()
                if pin:
                    return pin
    return _DEFAULT_GIT_PIN


GIT_PIN = _read_git_pin()
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
    """Return the installed work122 package or in-repo agent_comm_robustness suite."""
    cached = sys.modules.get(_ALIAS)
    if cached is not None and all(hasattr(cached, name) for name in _REQUIRED_EXPORTS):
        return cached

    repo_init = (Path(__file__).resolve().parent / "agent_comm_robustness" / "__init__.py").resolve()
    init_path = repo_init
    try:
        dist = importlib.metadata.distribution(DISTRIBUTION_NAME)
        cand = _distribution_init(dist)
        if cand.is_file():
            init_path = cand
    except importlib.metadata.PackageNotFoundError as err:
        if not repo_init.is_file():
            raise ImportError(
                f"The {DISTRIBUTION_NAME} distribution is not installed. {install_hint()}"
            ) from err

    spec = importlib.util.spec_from_file_location(
        _ALIAS,
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the wire suite from {init_path}")
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
    """Installed distribution version of the wire suite."""
    load_suite()
    try:
        return importlib.metadata.version(DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError:
        return "0.2.0"


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
