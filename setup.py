"""Install CirrusMachine together with its preserved scattering backend."""
from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).parent
setup(
    name="cirrusmachine",
    version="0.1.0",
    description="Statistical synthesis of Galactic cirrus images",
    long_description=(ROOT / "README.md").read_text(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"scattering": ["README.md"]},
    install_requires=[
        "numpy>=1.24", "scipy>=1.10", "astropy>=5.3",
        "matplotlib>=3.7", "torch>=2.1", "PyYAML>=6",
        "tqdm>=4.65", "appdirs>=1.4", "photutils>=1.9", "reproject>=0.12",
    ],
    extras_require={
        "dev": ["pytest>=7"],
        "notebooks": ["jupyterlab>=4", "ipykernel>=6"],
        "astronomy": ["healpy>=1.16", "turbustat>=1.3", "maskfill"],
    },
)
