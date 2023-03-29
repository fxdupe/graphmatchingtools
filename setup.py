from setuptools import setup
from setuptools import find_packages

setup(
    scripts=[
        "scripts/KMeans_Example.py",
        "scripts/PyTorchGeo_Example.py",
        "scripts/RandomGraphMatching.py",
        "scripts/INTGraph_Example.py",
    ],
    packages=find_packages(where=".", exclude=("tests*",)),
)
