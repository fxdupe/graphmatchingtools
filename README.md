# Graph Matching Tools

This toolbox propose a common ground for several graph matching methods. 
We propose our methods with a set of state-of-art methods using our own implementation.
The scope of this work is to offer reproductible research and . 

## Current available methods

### Pairwise method

- [KerGM](https://papers.nips.cc/paper/2019/hash/cd63a3eec3319fd9c84c942a08316e00-Abstract.html)

### Multiway methods (state-of-art)

- [HiPPI](https://openaccess.thecvf.com/content_ICCV_2019/html/Bernard_HiPPI_Higher-Order_Projected_Power_Iterations_for_Scalable_Multi-Matching_ICCV_2019_paper.html)
- KMeans (as a naive way of doing the multimatch)
- [MSync](https://papers.nips.cc/paper/2013/hash/3df1d4b96d8976ff5986393e8767f5b2-Abstract.html)
- [MatchEIG](https://openaccess.thecvf.com/content_iccv_2017/html/Maset_Practical_and_Efficient_ICCV_2017_paper.html)
- [QuickMatch](https://openaccess.thecvf.com/content_iccv_2017/html/Tron_Fast_Multi-Image_Matching_ICCV_2017_paper.html)
- [Sparse Quadratic Optimisation over the Stiefel Manifold with Application to Permutation Synchronisation](https://openreview.net/forum?id=sl_0rQmHxQk)
- [IRGCL](https://papers.nips.cc/paper/2020/hash/ae06fbdc519bddaa88aa1b24bace4500-Abstract.html)

### Multiway methods (homemade)

- Kernelized multi-graph matching

### Mean graph method

- [MMM method based on pairwise matching](https://www.sciencedirect.com/science/article/abs/pii/S003132031630139X)

## Installation

The package can be installed in editable mode using the following command in the same repertory as *setup.py*,
```
pip install -e .
```

We also propose a configuration for [Poetry](https://python-poetry.org) as an alternative for installation.

## Examples

We provide 3 examples based on this toolbox,
1. KMeans for graph matching.
2. Application of our method on random graph with a comparison against MatchEIG.
3. Application of different methods on Willow and PascalVOC databases using
[Pytorch-Geometrics](https://pytorch-geometric.readthedocs.io/). For example to run
the MKerGM method on the *duck* class from Willow we can execute,
```
python scripts/PyTorchGeo_Example.py --category duck --sigma 70.0 --gamma 0.01  --rff 200 --iterations 20 --rank 10
```

These examples may require modules that are not required in the setup.

## Authors
- Guillaume Auzias (INT)
- François-Xavier Dupé (LIS)
- Sylvain Takerkart (INT)
- Rohit Yadav (INT, LIS)

All authors are from [Aix-Marseille University](univ-amu.fr).
