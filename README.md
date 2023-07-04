# Mist - A PyTorch Mutual information Estimation toolkit

[![PyPI version](https://badge.fury.io/py/torch-mist.svg)](https://badge.fury.io/py/torch-mist)
[![codecov](https://codecov.io/gh/mfederici/torch-mist/badge.svg)](https://codecov.io/gh/mfederici/torch-mist)
[![Documentation Status](https://readthedocs.org/projects/torch-mist/badge/?version=latest)](https://torch-mist.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

<img src="docs/logo.png" onerror="this.onerror=null" width="200">


Mutual Information Estimation toolkit based on pytorch. TO BE RELEASED SOON

## Installation

The package can be installed via pip as follows:
```bash
$ pip install torch_mist
```

## Usage
The `torch_mist` package provides the basic functionalities for sample-based continuous mutual information estimation using modern
neural network architectures.

Here we provide a simple example of how to use the package to estimate mutual information between pairs
of observations using the MINE estimator [[2]](#references).

First, we need to import and instantiate the estimator from the package:
```python
from torch_mist.estimators import mine

# Defining the estimator
estimator = mine(
    x_dim=1,                    # dimension of x
    y_dim=1,                    # dimension of y   
    hidden_dims=[32, 64, 32],   # hidden dimensions of the neural networks
)
```
then we can train the estimator:
```python
from torch_mist.utils import optimize_mi_estimator


train_log = optimize_mi_estimator(
    estimator=estimator,        # the estimator to train
    dataloader=dataloader,      # the dataloader returning pairs of x and y
    epochs=10,                  # the number of epochs
    device="cpu",               # the device to use
    return_log=True,            # whether to return the training log
)
```
Lastly, we can use the trained estimator to estimate the mutual information between pairs of observations:
```python
from torch_mist.utils import estimate_mi
value, std = estimate_mi(
    estimator=estimator,        # the estimator to use
    dataloader=dataloader,      # the dataloader returning pairs of x and y
    device="cpu",               # the device to use
)

print(f"Estimated MI: {value} +- {std}")
```

Please refer to the [documentation](https://torch-mist.readthedocs.io/en/latest/) for a detailed description of the package and its usage.




### Estimators
The basic estimators implemented in this package are summarized in the following table:

| Estimator                                     | Type                  | Models                                    |
|-----------------------------------------------|-----------------------|-------------------------------------------|
| NWJ [[1]](#references)                        | Discriminative        | $f_\phi(x,y)$                              |
| MINE  [[2]](#references)                      | Discriminative        | $f_\phi(x,y)$                              |
| InfoNCE [[3]](#references)                    | Discriminative        | $f_\phi(x,y)$                              |
| TUBA  [[4]](#references)                      | Discriminative        | $f_\phi(x,y)$, $b_\xi(x)$                  | 
| AlphaTUBA [[4]](#references)                  | Discriminative        | $f_\phi(x,y)$, $b_\xi(x)$                  |
| JS [[5]](#references)                         | Discriminative        | $f_\phi(x,y)$                              |
| SMILE [[6]](#references)                      | Discriminative        | $f_\phi(x,y)$                              |
| FLO [[7]](#references)                        | Discriminative        | $f_\phi(x,y)$, $b_\xi(x,y)$               | 
| BA [[8]](#references)                         | Generative            | $q_\theta(y\|x)$                          |          
| DoE [[9]](#references)                        | Generative            | $q_\theta(y\|x)$, $q_\psi(y)$             | 
| GM [[6]](#references)                         | Generative            | $q_\theta(x,y)$, $q_\psi(x)$, $r_\psi(y)$ |
| L1OUT [[4]](#references) [[10]](#references)  | Generative            | $q_\theta(y\|x)$                          |                  
| CLUB [[10]](#references)                      | Generative            | $q_\theta(y\|x)$                          |
| Discrete [[]](#references)                    | Generative (Discrete) | $Q(x)$, $Q(y)$                            |
| PQ [[11]](#references)                        | Generative (Discrete) | $Q(y)$, $q_\theta(Q(y)\|x)$               |

in which:
- $f_\phi(x,y)$ is a `critic` neural network with parameters $\phi, which maps pairs of observations to a scalar value.
Critics can be either `joint` or `separable` depending on whether they parametrize function of both $x$ and $y$ directly, 
or through the product of separate projection heads ( $f_\phi(x,y)=h_\phi(x)^T h_\phi(y)$ ) respectively.
- $b_\xi(x)$ is a `baseline` neural network with parameters $\xi$, which maps observations (or paris of observations) to a scalar value.
When the baseline is a function of both $x$ and $y$ it is referred to as a `joint_baseline`.
- $q_\theta(y\|x)$ is a conditional variational distribution `q_Y_given_X` used to approximate $p(y\|x)$ with parameters $\theta$.
Conditional distributions may have learnable parameters $\theta$ that are usually parametrized by a (conditional) normalizing flow.
- $q_\psi(y)$ is a marginal variational distribution `q_Y` used to approximate $p(y)$ with parameters $\psi$.
Marginal distributions may have learnable parameters $\psi$ that are usually parametrized by a normalizing flow.
- $q_\theta(x,y)$ is a joint variational distribution `q_XY` used to approximate $p(x,y)$ with parameters $\theta$.
Joint distributions may have learnable parameters $\theta$ that are usually parametrized by a normalizing flow.
- $Q(x)$ and $Q(y)$ are `quantization` functions that map observations to a finite set of discrete values.

#### Hybrid estimators
The `torch_mist` package allows to combine Generative and Discriminative estimators in a single hybrid estimators as proposed in [[11]](#references)[[12]](#references).


### References

[[1] ](https://arxiv.org/abs/0809.0853) Nguyen, XuanLong, Martin J. Wainwright, and Michael I. Jordan. "Estimating divergence functionals and the likelihood ratio by convex risk minimization." IEEE Transactions on Information Theory 56.11 (2010): 5847-5861.

[[2]](https://arxiv.org/abs/1801.04062) Belghazi, Mohamed Ishmael, et al. "Mutual information neural estimation." International conference on machine learning. PMLR, 2018.

[[3]](https://arxiv.org/abs/1807.03748) Oord, Aaron van den, Yazhe Li, and Oriol Vinyals. "Representation learning with contrastive predictive coding." arXiv preprint arXiv:1807.03748 (2018).

[[4]](https://arxiv.org/abs/1905.06922)  Poole, Ben, et al. "On variational bounds of mutual information." International Conference on Machine Learning. PMLR, 2019.

[[5]](https://arxiv.org/abs/1808.06670) Hjelm, R. Devon, et al. "Learning deep representations by mutual information estimation and maximization." arXiv preprint arXiv:1808.06670 (2018).

[[6]](https://arxiv.org/abs/1910.06222) Song, Jiaming, and Stefano Ermon. "Understanding the limitations of variational mutual information estimators." arXiv preprint arXiv:1910.06222 (2019).

[[7]](https://arxiv.org/abs/2107.01131) Guo, Qing, et al. "Tight mutual information estimation with contrastive fenchel-legendre optimization." Advances in Neural Information Processing Systems 35 (2022): 28319-28334.

[[8]](https://aivalley.com/Papers/MI_NIPS_final.pdf) Barber, David, and Felix Agakov. "The im algorithm: a variational approach to information maximization." Advances in neural information processing systems 16.320 (2004): 201.

[[9]](https://arxiv.org/abs/1811.04251) McAllester, David, and Karl Stratos. "Formal limitations on the measurement of mutual information." International Conference on Artificial Intelligence and Statistics. PMLR, 2020.

[[10]](https://arxiv.org/abs/2006.12013) Cheng, Pengyu, et al. "Club: A contrastive log-ratio upper bound of mutual information." International conference on machine learning. PMLR, 2020.

[[11]](https://arxiv.org/abs/2306.00608) Federici, Marco, David Ruhe, and Patrick Forré. "On the Effectiveness of Hybrid Mutual Information Estimation." arXiv preprint arXiv:2306.00608 (2023).

[[12]](https://arxiv.org/abs/2303.06992) Brekelmans, Rob, et al. "Improving mutual information estimation with annealed and energy-based bounds." arXiv preprint arXiv:2303.06992 (2023).

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`torch_mist` was created by Marco Federici. It is licensed under the terms of the MIT license.

## Credits

`torch_mist` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
