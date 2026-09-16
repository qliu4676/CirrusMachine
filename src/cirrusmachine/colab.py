"""Colab setup helpers for CirrusMachine notebooks.

The original ``setup_scattering_colab.py`` remains usable.  This module offers
the same setup steps behind configurable functions so reusable science modules
do not need hard-coded ``/content/drive`` paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Optional


DEFAULT_COLAB_PROJECT = str(Path(__file__).resolve().parents[2])


@dataclass(frozen=True)
class ColabSetupConfig:
    """Paths and branch used to configure the bundled scattering package."""

    project_root: Path
    scattering_repo: Path
    requirements_file: Path
    branch: str = "optim"

    @classmethod
    def from_defaults(
        cls,
        project_root: str | os.PathLike[str] | None = None,
        branch: Optional[str] = None,
    ) -> "ColabSetupConfig":
        root = Path(
            project_root
            or os.environ.get("CIRRUSMACHINE_COLAB_PROJECT")
            or DEFAULT_COLAB_PROJECT
        )
        return cls(
            project_root=root,
            scattering_repo=root / "src",
            requirements_file=root / "requirements.txt",
            branch=branch or os.environ.get("CIRRUSMACHINE_SCATTERING_BRANCH", "optim"),
        )


def run_command(cmd: Iterable[str]) -> None:
    """Run a setup command with the same visible logging as the old script."""

    cmd_list = [str(part) for part in cmd]
    print(">", " ".join(cmd_list))
    subprocess.check_call(cmd_list)


def ensure_scattering_repo(config: ColabSetupConfig) -> None:
    """Fetch, checkout, and pull the configured scattering repository branch."""

    if not config.scattering_repo.is_dir():
        raise FileNotFoundError(f"Repo not found: {config.scattering_repo}")
    raise RuntimeError("The scattering backend is bundled. Update the CirrusMachine checkout explicitly.")


def install_requirements(config: ColabSetupConfig) -> None:
    """Install Colab requirements for notebooks."""

    run_command([sys.executable, "-m", "pip", "install", "-q", "-r", config.requirements_file])


def add_scattering_to_path(config: ColabSetupConfig) -> None:
    """Make the local ``scattering_transform`` checkout importable."""

    repo = str(config.scattering_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def install_scattering_editable(config: ColabSetupConfig) -> None:
    """Try editable installation of the bundled scattering checkout."""

    run_command([sys.executable, "-m", "pip", "install", "-q", "-e", config.project_root])


def setup_colab_environment(
    config: ColabSetupConfig | None = None,
    *,
    update_repo: bool = False,
    install_packages: bool = True,
) -> ColabSetupConfig:
    """Configure Colab for the current notebooks and return the used config."""

    cfg = config or ColabSetupConfig.from_defaults()
    if update_repo:
        ensure_scattering_repo(cfg)
    if install_packages:
        install_requirements(cfg)
    try:
        install_scattering_editable(cfg)
    except Exception as exc:
        print("Editable install failed, falling back to sys.path only.")
        print("Reason:", exc)
        add_scattering_to_path(cfg)
    return cfg


def import_common_colab_packages() -> dict[str, object]:
    """Import common packages used repeatedly by current Colab notebooks."""

    import math
    import numpy as np
    import torch
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt
    import astropy.units as u
    from astropy.io import fits
    from astropy.nddata import Cutout2D, block_reduce
    from astropy.table import Table
    from astropy.modeling import models
    from astropy.coordinates import SkyCoord
    from astropy.convolution import convolve_fft, Gaussian2DKernel
    import scattering

    plt.rcParams["image.origin"] = "lower"
    return locals()
