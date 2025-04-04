# Graph Matching Tools

This toolbox proposes a common ground for several graph matching methods.
We propose our methods with a set of state-of-art methods using our own implementation.
The scope of this work is to offer reproducible research and alternatives.

## Current available methods

### Pairwise method

- [KerGM](https://papers.nips.cc/paper/2019/hash/cd63a3eec3319fd9c84c942a08316e00-Abstract.html)
- [GWL](https://proceedings.mlr.press/v97/xu19b.html)
- [FGW](https://proceedings.mlr.press/v97/titouan19a.html)
- [RRWM](https://link.springer.com/chapter/10.1007/978-3-642-15555-0_36)
- [Factorized RRWM](https://www.sciencedirect.com/science/article/abs/pii/S0031320323002984)
- [GRASP](https://dl.acm.org/doi/abs/10.1145/3561058)

### Multiway methods (state-of-art)

- [HiPPI](https://openaccess.thecvf.com/content_ICCV_2019/html/Bernard_HiPPI_Higher-Order_Projected_Power_Iterations_for_Scalable_Multi-Matching_ICCV_2019_paper.html)
- KMeans (as a naive way of doing the multimatch)
- [MSync](https://papers.nips.cc/paper/2013/hash/3df1d4b96d8976ff5986393e8767f5b2-Abstract.html)
- [MatchEIG](https://openaccess.thecvf.com/content_iccv_2017/html/Maset_Practical_and_Efficient_ICCV_2017_paper.html)
- [QuickMatch](https://openaccess.thecvf.com/content_iccv_2017/html/Tron_Fast_Multi-Image_Matching_ICCV_2017_paper.html)
- [Sparse Quadratic Optimisation over the Stiefel Manifold with Application to Permutation Synchronisation](https://openreview.net/forum?id=sl_0rQmHxQk)
- [IRGCL](https://papers.nips.cc/paper/2020/hash/ae06fbdc519bddaa88aa1b24bace4500-Abstract.html)
- [GA-MGMC](https://proceedings.neurips.cc/paper/2020/hash/e6384711491713d29bc63fc5eeb5ba4f-Abstract.html)
- [MatchALS](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Zhou_Multi-Image_Matching_via_ICCV_2015_paper.html)
- [Symmetric Sparse Boolean Matrix Factorization](https://arxiv.org/abs/2102.01570)
- [Factorized multi-graph matching](https://www.sciencedirect.com/science/article/abs/pii/S0031320323002984)
- [MIXER: Multiattribute, Multiway Fusion of Uncertain Pairwise Affinities](https://ieeexplore.ieee.org/abstract/document/10058986/)

### Multiway method (homemade)

- [Kernelized multi-graph matching](https://hal.science/hal-03809028v1)

### Mean graph methods

- [MMM method based on pairwise matching](https://www.sciencedirect.com/science/article/abs/pii/S003132031630139X)
- [Wasserstein barycenter using FGW](https://proceedings.mlr.press/v97/titouan19a.html)

## Auxiliary methods (aka solvers...)

### Optimal transport

- [Sinkhorn-Knopp](https://proceedings.neurips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html)
- [Sinkhorn algorithm with sparse Newton iterations](https://arxiv.org/abs/2401.12253)

### Manifold optimization

- [Fast and accurate optimization on the orthogonal manifold without retraction](https://proceedings.mlr.press/v151/ablin22a)

### Graph generation

- Sucal pits graph using methods and code from papers [1](https://www.sciencedirect.com/science/article/pii/S1361841516300251),
[2](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0293886) and [3](https://ieeexplore.ieee.org/abstract/document/9897185/).

## Installation

The package can be installed in editable mode using the following command from the base directory,
```shell
pip install -e .
```

We also propose a configuration for [Poetry](https://python-poetry.org) as an alternative for installation.

The documentation can be build using *sphinx-build* (please install the required packages for development). For example
to generate the HTML documentation you can use,
```shell
sphinx-build -b html docs <output>
```
where <output> is the output directory for documentation.

## Examples

We provide several examples based on this toolbox,
1. KMeans for graph matching.
2. Application of our method on random graph with a comparison against MatchEIG.
3. Application of different methods on Willow and PascalVOC databases using
[Pytorch-Geometrics](https://pytorch-geometric.readthedocs.io/). For example to run
the MKerGM method on the *duck* class from Willow we can execute,
```shell
gmt_demo_pytorchdata --category duck --sigma 70.0 --gamma 0.01  --rff 200 --iterations 20 --rank 10
```
4. Generation of simulated sucal pits graph. For example, to generate 11 graphs (one reference with 10 noisy version)
and kappa=400, we can execute,
```shell
graph_matching_tools/demos/gmt_demo_generate_graph.py --add_outliers --suppress_nodes --coord_noise_kappa 400 --sample_number 10 --save
```

These examples may require modules that are not required in the setup.

## Current Authors
- Guillaume Auzias (INT)
- François-Xavier Dupé (LIS)
- Sylvain Takerkart (INT)

All authors are from [Aix-Marseille University](https://univ-amu.fr).

## Previous Authors
- Rohit Yadav (INT, LIS)
- Marius Thorre (LIS)
