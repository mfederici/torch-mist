# Mist - A PyTorch Mutual information Estimation toolkit
[![arXiv](https://img.shields.io/badge/arXiv-2306.00608-b31b1b.svg)](https://arxiv.org/abs/2306.00608)
[![PyPI version](https://badge.fury.io/py/torch-mist.svg)](https://badge.fury.io/py/torch-mist)
![Build workflow](https://github.com/mfederici/torch-mist/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/mfederici/torch-mist/badge.svg)](https://codecov.io/gh/mfederici/torch-mist)
[![Documentation Status](https://readthedocs.org/projects/torch-mist/badge/?version=latest)](https://torch-mist.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img src="https://github.com/mfederici/torch-mist/blob/main/docs/_static/logo.png?raw=true" onerror="this.onerror=null" width="200">


Mutual Information Estimation toolkit based on pytorch. 
Please refer to the [documentation](https://torch-mist.readthedocs.io/en/latest/index.html) for additional details regarding
[installation](https://torch-mist.readthedocs.io/en/latest/notebooks/installation.html), [usage](https://torch-mist.readthedocs.io/en/latest/notebooks/usage.html),
tutorials and pracical use-case example.


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
Consider the variables $x$ and $y$ as of shape `[N, x_dim]`, `[N, y_dim]` respectively sampled from some joint distribution $p(x,y)$.
Mutual information can be estimated directly using the `estimate_mi` utility function that takes care of fitting the estimator's parameters and evaluating mutual information.

```python3
from torch_mist import estimate_mi
from sklearn.datasets import load_iris

# Load the Iris Dataset as a pandas DataFrame
iris_dataset = load_iris(as_frame=True)['data']

# Estimate how much information the petal length and its width have in common
estimated_mi, estimator, train_log = estimate_mi(
    data=iris_dataset,          # The dataset (as a pandas.DataFrame, many other formats are supported)
    x_key='petal length (cm)',  # Consider the 'petal length' column as x
    y_key='petal width (cm)',   # And the 'petal witdh` as y
    estimator_name='mine',      # Use the MINE mutual information estimator
    max_iterations=1000,        # Number of maximum train iterations 
)

print(f"Mutual information estimated value: {estimated_mi} nats")
```
The `estimate_mi` function supports additional data formats such as tuples `(x, y)` or dictionaries `{'x':x, 'y':y}` 
of tensors, but also datasets (see `torch.utils.data.Dataset`) and torch `DataLoaders`.

Additional flags can be used to customize the estimators, training and evaluation procedure, as detailed in the [documentation](https://torch-mist.readthedocs.io/en/latest).

## Command line
The `torch_mist` package provides basic functionality to estimate mutual information directly from the command line.
Given a file `iris.csv` containing the columns `(sepal_1, sepal_2, petal_1, petal_2)`, one can estimate mutual information between
the `sepal` and `petal` 2-dimensional features with:
```bash
mist data=csv data.filepath=iris.csv mi_estimator=js x_key=sepal y_key=petal
```
The same flags and options provided by the `estimate_mi` function are also available from command line.

Additionally, internal properties of the estimator can also be easily specified:
```bash
mist data=csv data.filepath=iris.csv mi_estimator=js x_key=sepal y_key=petal \ 
  # Train on GPU
  device=cuda \
  # Use AdamW for the optimization
  estimation.optimizer_class._target_=torch.optim.AdamW \
  # Use ELU as nonlinearities
  +mi_estimator.nonlinearity=torch.nn.ELU \
  # Change the batch size to 256
  params.batch_size=256
  # Log on weights and bias
  logger=wandb
```

To visualize the full list use:
```bash
mist data=csv --help
```
The `mist` CLI is implemented using [hydra](https://hydra.cc/) and the full configuration can be accessed [here](scripts/config).


## Advanced usage
It is possible to manually instantiate, train and evaluate the mutual information estimators.

```python3
from torch_mist.estimators import mine
from torch_mist.utils.train import train_mi_estimator
from torch_mist.utils import evaluate_mi

# Instantiate the JS mutual information estimator
estimator = mine(
    x_dim=1,
    y_dim=1,
    hidden_dims=[64, 32],
)

# Define x and y as the vectors of petal lengths and widths, respectively
x = iris_dataset['petal length (cm)'].values 
y = iris_dataset['petal width (cm)'].values

# Train it on the given samples
train_log = train_mi_estimator(
    estimator=estimator,
    data=(x, y),
    batch_size=64,
    max_iterations=1000
)

# Evaluate the estimator on the entirety of the data
estimated_mi = evaluate_mi(
    estimator=estimator,
    data=(x, y),
    batch_size=64
)

print(f"Mutual information estimated value: {estimated_mi} nats")
```
Note that the two code snippets above perform the same procedure.
Please refer to the [documentation](https://torch-mist.readthedocs.io/en/latest/) for a detailed description of the package and its usage.


### Estimators
Each estimator implemented in the library is an instance of `MutualInformationEstimator` and can be instantiated
through a simplified utility functions
```python3
############################
# Simplified instantiation #
############################
from torch_mist.estimators import mine

estimator = mine(
    x_dim=1,
    y_dim=1,
    neg_samples=16,
    hidden_dims=[64, 32],
    critic_type='joint'
)
```
or directly using the corresponding `MutualInformationEstimator` class

```python3
##########################
# Advanced instantiation #
##########################
from torch_mist.estimators import MINE
from torch_mist.critic import JointCritic
from torch import nn

# First we define the critic architecture
critic = JointCritic(  # Wrapper to concatenate the inputs x and y 
    joint_net=nn.Sequential(  # The neural network architectures that maps [x,y] to a scalar
        nn.Linear(x.shape[-1] + y.shape[-1], 32),
        nn.ReLU(True),
        nn.Linear(32, 32),
        nn.ReLU(True),
        nn.Linear(32, 1)
    )
)

# Then we pass it to the MINE constructor
estimator = MINE(
    critic=critic,
    neg_samples=16,
)
```
Note that the simplified and advanced instantiation reported in the example above result in the same model.




The basic estimators implemented in this package are summarized in the following table:

| Estimator                                             | Type                     | Models                                    | Hyperparameters   | 
|-------------------------------------------------------|--------------------------|-------------------------------------------|-------------------|
| NWJ [[1]](#references)                                | Discriminative           | $f_\phi(x,y)$                             | M                 | 
| MINE  [[2]](#references)                              | Discriminative           | $f_\phi(x,y)$                             | M, $\gamma_{EMA}$ | 
| InfoNCE [[3]](#references)                            | Discriminative           | $f_\phi(x,y)$                             | M                 | 
| TUBA  [[4]](#references)                              | Discriminative           | $f_\phi(x,y)$, $b_\xi(x)$                 | M                 | 
| AlphaTUBA [[4]](#references)                          | Discriminative           | $f_\phi(x,y)$, $b_\xi(x)$                 | M, $\alpha$       | 
| JS [[5]](#references)                                 | Discriminative           | $f_\phi(x,y)$                             | M                 | 
| SMILE [[6]](#references)                              | Discriminative           | $f_\phi(x,y)$                             | M, $\tau$         |
| FLO [[7]](#references)                                | Discriminative           | $f_\phi(x,y)$, $b_\xi(x,y)$               | M                 |
| BA [[8]](#references)                                 | Generative               | $q_\theta(y\|x)$                          | -                 |         
| DoE [[9]](#references)                                | Generative               | $q_\theta(y\|x)$, $q_\psi(y)$             | -                 |
| GM [[6]](#references)                                 | Generative               | $q_\theta(x,y)$, $q_\psi(x)$, $q_\psi(y)$ | -                 |
| L1OUT [[4]](#references) [[10]](#references)          | Generative               | $q_\theta(y\|x)$                          | -                 |                 
| CLUB [[10]](#references)                              | Generative               | $q_\theta(y\|x)$                          | -                 |
| Binned [[13]](https://arxiv.org/abs/cond-mat/0305641) | Transformed (Generative) | $Q(x)$, $Q(y)$                            | -                 |
| PQ [[11]](#references)                                | Transformed (Generative) | $Q(y)$, $q_\theta(Q(y)\|x)$               | -                 |

in which the following models are used:
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

And the following hyperparameters:
- $M \in [1, N]$ is the number of samples used to estimate the log-normalization constant for each element in the batch.
- $\gamma_{EMA} \in (0,1]$ is the exponential moving average decay used to update the baseline in MINE.
- $\alpha \in [0,1]$ is the weight of the baseline in AlphaTUBA (0 corresponds to InfoNCE, 1 to TUBA).
- $\tau \in [0..]$ is used to define the interval $[-\tau,\tau]$ in which critic values are clipped in SMILE.

#### Hybrid estimators
The `torch_mist` package allows to combine Generative and Discriminative estimators in a single hybrid estimators as proposed in [[11]](#references)[[12]](#references).
 Hybrid mutual information estimators combine the flexibility of discriminative mutual information estimators with the lower 
variance of generative estimators. 
```python
from torch_mist.estimators.hybrid import ResampledHybridMIEstimator
from torch_mist.estimators import js, doe

# Use the proposal r(y|x) to sample negatives instead of p(y)
estimator = ResampledHybridMIEstimator(
    # Difference of Entropies generative estimator
    generative_estimator=doe(
        x_dim=x.shape[-1],
        y_dim=y.shape[-1],
        hidden_dims=[32, 32],
    ),
    # NWJ discriminative estimator
    discriminative_estimator=js(
        x_dim=x.shape[-1],
        y_dim=y.shape[-1],
        hidden_dims=[32, 32],
        neg_samples=16
    )
)

```
Further details on the available hybrid mutual information estimators and additional details are reported in the 
[tutorial](https://torch-mist.readthedocs.io/en/latest/notebooks/hybrid.html#) available in the [documentation](https://torch-mist.readthedocs.io/en/latest/index.html).


### Training and Evaluation
Most of the estimators included in this package are parametric and require a training procedure for accurate estimation.
The `train_mi_estimator` utility function supports either row data `x` and `y` as `numpy.array` or `torch.Tensor`.

```python3
from torch_mist.utils.train import train_mi_estimator

######################################
# Training using tensors for x and y #
######################################
# By default 10% of the data is used for cross-validation and early stopping
train_log = train_mi_estimator(
    estimator=estimator,
    data=(x,y),
    batch_size=64,
    valid_percentage=0.1,
)
```
Alternatively, it is possible to use a `torch.utils.DataLoader` that returns eiter batches of pairs `(batch_x, batch_y)`
or dictionaries of batches `{'x': batch_x, 'y': batch_y}`, with `batch_x` of shape `[batch_size, ..., x_dim]` and `[batch_size, ..., y_dim]` respectively.
```python3
#############################
# Training with DataLoaders #
#############################
from torch_mist.utils.data import SampleDataset
from torch.utils.data import DataLoader, random_split

# We provide an utility to make the tensors into a torch.utils.data.Dataset object
# This can be replaced with any other Dataset object that may load the data from disk
dataset = SampleDataset(
    samples={'x': x, 'y': y}
)

# Split into train and validation
train_size = int(len(dataset)*0.9)
valid_size = len(dataset)-train_size
train_set, valid_set = random_split(dataset, [train_size, valid_size])

# Instantiate the dataloaders
train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True,
    num_workers=8
)

valid_loader = DataLoader(
    valid_set,
    batch_size=64,
    num_workers=8
)

# Train using the specified dataloaders
# Note that the validation set is optional but recommended to prevent overfitting.

train_log = train_mi_estimator(
    estimator=estimator,
    data=train_loader,
    valid_data=valid_loader,
)
```
The two options result in the same training procedure, but we recommend using `DataLoader` for larger datasets.

Both `DataLoader` and `torch.Tensor` (or `np.array`) can be used for the `evaluate_mi` function.


# References

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

[[13]](https://arxiv.org/abs/cond-mat/0305641) Kraskov, Alexander, Harald Stögbauer, and Peter Grassberger. "Estimating mutual information." Physical review E 69.6 (2004): 066138.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`torch_mist` was created by Marco Federici. It is licensed under the terms of the MIT license.

## Credits

`torch_mist` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
