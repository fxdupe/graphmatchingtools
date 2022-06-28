# Graph Matching Tools

Toolbox for common ground for graph matching methods

## Current available methods

### Pairwise method

- KerGM

### Multiway methods

- HiPPI
- KMeans
- MSync
- MatchEIG
- QuickMatch
- Sparse Quadratic Optimisation over the Stiefel Manifold with Application to Permutation Synchronisation
- IRGCL
- Multiway KerGM

### Mean graph method

- MM method proposed by Jain

## Installation

The package can be installed in editable mode using the following command in the same repertory as *setup.py*,
```
pip install -e .
```

We also propose a configuration for [Poetry](https://python-poetry.org) as an alternative for installation.

## Examples

We provide 2 examples based on this toolbox,
1. KMeans for graph matching.
2. Application of different methods on Willow and PascalVOC databases using
[Pytorch-Geometrics](https://pytorch-geometric.readthedocs.io/). For example,
```
python scripts/PyTorchGeo_Example.py --category duck --sigma 70.0 --gamma 0.01  --rff 200
```
will run the MKerGM method on the *duck* class from Willow.

These examples may require modules that are not required in the setup.

## Authors
- Guillaume Auzias (INT)
- François-Xavier Dupé (LIS)
- Sylvain Takerkart (INT)
- Rohit Yadav (INT, LIS)

All authors are from [Aix-Marseille University](univ-amu.fr).
