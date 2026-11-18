# Comparator sources

## ALMA

`model/alma.py` retains the alternating updates from the Python implementation
supplied with this BALM project. Its algorithm is described by Xing Fan,
Marianna Pensky, Feng Yu and Teng Zhang, **ALMA: Alternating Minimization
Algorithm for Clustering Mixture Multilayer Network**, Journal of Machine
Learning Research 23(330), 1–46, 2022.

Paper: https://jmlr.org/papers/v23/21-0182.html

The training-only missing-entry imputation and final loading-row clustering
are the adapters used in the revised BALM experiments. ALMA loading factors
are not BALM simplex memberships. This file is a project implementation of
the published algorithm; it is not represented as an upstream ALMA package.

## Tensor block model (TBM)

`model/tbm.py` and `model/tbm.R` call the separately installed R package
**tensorsparse 3.0**, by Miaoyan Wang and Yuchen Zeng, under **GPL (>= 2)**.
No tensorsparse implementation is vendored in this repository. Its license
and copyright notices are distributed with that package.

Package: https://CRAN.R-project.org/package=tensorsparse

Method: Miaoyan Wang and Yuchen Zeng, **Multiway Clustering via Tensor Block
Models**, Advances in Neural Information Processing Systems 32, 2019.

The adapter uses unpenalized fitting, separate partitions for the two node
modes, training-only imputation and averaging of the two predicted edge
directions. It does not substitute the earlier Python symmetric masked
least-squares comparator for tensorsparse.

These dependency notices do not assign a license to BALM project code.
