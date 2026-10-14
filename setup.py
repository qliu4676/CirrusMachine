"""Install CirrusMachine with its pinned upstream scattering source dependency."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
SCATTERING_SOURCE = Path("external/scattering_transform/scattering")
if not (ROOT / SCATTERING_SOURCE / "Scattering2d.py").is_file():
    raise RuntimeError(
        "Initialize the scattering source dependency first: "
        "git submodule update --init external/scattering_transform"
    )
setup(
    name="cirrusmachine",
    version="0.1.0",
    description="Statistical synthesis of Galactic cirrus images",
    long_description=(ROOT / "README.md").read_text(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(
        "src", exclude=["cirrusmachine.legacy", "cirrusmachine.legacy.*"]
    )
    + find_packages(
        "external/scattering_transform", include=["scattering", "scattering.*"]
    ),
    package_dir={"": "src", "scattering": str(SCATTERING_SOURCE)},
    project_urls={
        "Scattering source": "https://github.com/SihaoCheng/scattering_transform/tree/04f36a64677c038538d34828430518cf749e77df",
    },
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "astropy>=5.3",
        "matplotlib>=3.7",
        "torch>=2.1",
        "PyYAML>=6",
        "tqdm>=4.65",
        "appdirs>=1.4",
        "photutils>=1.9",
    ],
    extras_require={
        "tests": ["pytest>=7"],
        "examples": ["jupyterlab>=4", "ipykernel>=6", "ipywidgets>=8"],
    },
)
