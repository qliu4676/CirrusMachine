"""Install CirrusMachine and its runtime dependencies."""

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
    packages=find_packages(
        "src", exclude=["cirrusmachine.legacy", "cirrusmachine.legacy.*"]
    ),
    package_dir={"": "src"},
    package_data={"cirrusmachine._vendor.scattering": ["UPSTREAM.json"]},
    license_files=["LICENSE", "THIRD_PARTY_NOTICES"],
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
        "photutils>=1.9",
    ],
    extras_require={
        "tests": ["pytest>=7"],
        "examples": ["jupyterlab>=4", "ipykernel>=6"],
    },
)
