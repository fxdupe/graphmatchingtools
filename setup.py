from setuptools import setup
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="GraphMatchingTools",
    version="1.0.0",
    description="Graph Matching Tools",
    author="François-Xavier Dupé",
    author_email="francois-xavier.dupe@lis-lab.fr",
    url="https://github.com/fxdupe/graphmatchingtools",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=[
      "scripts/KMeans_Example.py",
    ],
    packages=find_packages(where=".", exclude=("tests",)),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "networkx",
        "scikit-learn",
    ],
    python_requires=">=3.8",
)
