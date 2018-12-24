## L0-Motivated Low-Rank Sparse Subspace (LRSSC)

## Overview

In this repository we provide MATLAB implementations of GMC-LRSSC and L0-LRSSC proposed in [L0-Motivated Low-Rank Sparse Subspace Clustering](https://ieeexplore.ieee.org/document/8573150). GMC-LRSSC solves subspace clustering problem by using regularization based on multivariate generalization of the minimax-concave (GMC) penalty function, while L0-LRSSC solves the Schatten-0 and L0 quasi-norm regularized objective.
To run proposed algorithms, we provide some example scripts and data.

## Datasets

The datasets used in the paper can be found in the 'datasets' directory. Datasets directory includes Extended Yale B dataset from [http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html](http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html), the USPS dataset from [https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps), the MNIST dataset from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/) and ISOLET1 dataset from the UCI Machine Learning Repository ([https://archive.ics.uci.edu/ml/datasets/isolet](https://archive.ics.uci.edu/ml/datasets/isolet)).

## Citing

When using the code in your research work, please cite "Multi-view Low-rank Sparse Subspace Clustering" by Maria Brbic and Ivica Kopriva.

    @article{brbic2018,
    title={$\ell_0$-Motivated Low-Rank Sparse Subspace Clustering},
    author={Brbi\'c, Maria and Kopriva, Ivica},
    journal={IEEE Transactions on Cybernetics},
    year={2018},
    doi={10.1109/TCYB.2018.2883566}, 
    }

## Acknowledgements

This work was supported by the Croatian Science Foundation (Structured Decompositions of Empirical Data for Computationally-Assisted Diagnoses of Disease) under Grant IP-2016-06-5235, and by the European Regional Development Fund (DATACROSS) under Grant KK.01.1.1.01.0009.

