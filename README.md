# Mist - A PyTorch Mutual information Estimation toolkit

<img src="docs/logo.png" alt="alt text" width="200">



[![PyPI version](https://badge.fury.io/py/torch-mist.svg)](https://badge.fury.io/py/torch-mist)
[![codecov](https://codecov.io/gh/mfederici/torch-mist/branch/master/graph/badge.svg)](https://codecov.io/gh/mfederici/torch-mist)
[![Documentation Status](https://codecov.io/gh/mfederici/torch-mist/badge.svg)](https://torch-mist.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



Mutual Information Estimation toolkit based on pytorch. TO BE RELEASED SOON

## Installation

```bash
$ pip install torch_mist
```

## Usage

The estimators are reported in the following table

| Estimator                                                                              | Type           |
|----------------------------------------------------------------------------------------|----------------|
| NWJ [[1]](https://arxiv.org/abs/0809.0853)                                             | Discriminative |
| MINE  [[2]](https://arxiv.org/abs/1801.04062)                                          | Discriminative |
| InfoNCE [[3]](https://arxiv.org/abs/1807.03748)                                        | Discriminative |
| TUBA  [[4]](https://arxiv.org/abs/1905.06922)                                          | Discriminative | 
| AlphaTUBA [[4]](https://arxiv.org/abs/1905.06922)                                      | Discriminative |
| JS [[5]](https://arxiv.org/abs/1808.06670)                                             | Discriminative |
| SMILE [[6]](https://arxiv.org/abs/1910.06222)                                          | Discriminative |
| FLO [[7]](https://arxiv.org/abs/2107.01131)                                            | Discriminative |
| BA [[8]](https://aivalley.com/Papers/MI_NIPS_final.pdf)                                | Generative     |
| DoE [[9]](https://arxiv.org/pdf/1811.04251.pdf)                                        | Generative     |
| GM [[6]](https://arxiv.org/abs/1910.06222)                                             | Generative     |
| L1OUT [[4]](https://arxiv.org/abs/1905.06922) [[10]](https://arxiv.org/abs/2006.12013) | Generative     |
| CLUB [[10]](https://arxiv.org/abs/2006.12013)                                          | Generative     |
| GDE [[]]()                                                                             | Generative     |
| PQ [[11]](https://arxiv.org/abs/2306.00608)                                            | Generative     |
- TODO




## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`torch_mist` was created by Marco Federici. It is licensed under the terms of the MIT license.

## Credits

`torch_mist` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
