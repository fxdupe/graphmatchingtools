Changelog
=========

0.9.0
-----

New methods:

* Classical version of RRWM.
* GRASP (version without any trick).
* New utils: optimization on Group Orthogonal and Stiefel manifold.
* Update documentation.
* Update tools.


0.8.0
-----

Here the most important changes:

* Add the factorized multi-graph matching method.
* Add the factorized version of the RRWM pairwise matching.
* Update documentation.
* Update tools.


0.7.0
-----

This is the first version with a working documentation.
Here the most important changes:

* Add two graph pairwise methods based on optimal transport (FGW and GWL).
* Add two state-of-art multigraph methods: GA-MGM and MatchALS.
* Bug corrections.
* Begin Sphinx documentation with automatic API generation.
* Improve some of the unit tests.
* Update the requirements.
* Using Gitlab-CI for building wheel.

Note: there is now a dependency with **jax** but it is only require for GWL.
