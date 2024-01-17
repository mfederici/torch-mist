from torch.optim import AdamW
from torchvision.models import resnet18
from torch import nn
import torch

from torch_mist.distributions import (
    transformed_normal,
    conditional_transformed_normal,
)
from torch_mist.utils.freeze import freeze
from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torch_mist.utils import train_mi_estimator
from torch_mist.estimators import (
    TransformedMIEstimator,
    DoE,
    ResampledHybridMIEstimator,
    nwj,
    MIEstimator,
)
from torch_mist.estimators.multi import MultiMIEstimator
from torch_mist.utils.logging import PandasLogger

# Dimensionality of the resnet18 representation
Z_DIM = 512

# Landmark names
LEFT_EYE = "left_eye"
RIGHT_EYE = "right_eye"
NOSE = "nose"
RIGHT_MOUTH = "right_mouth"
LEFT_MOUTH = "left_mouth"
LANDMARKS = [LEFT_EYE, RIGHT_EYE, NOSE, LEFT_MOUTH, RIGHT_MOUTH]

# Representation names
RANDOM = "random_init"
PRETRAINED = "pretrained"
FINETUNED = "finetuned"
REPRESENTATIONS = [RANDOM, PRETRAINED, FINETUNED]


# We define a simple utility class for the resnet18 encoder
class ResNet18Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.resnet = resnet18(**kwargs)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def __repr__(self):
        return "ResNet18Encoder()"


# And wrap the original CelebA dataset to return a dictionary containing the image and each landmark
class CelebALandmarks(CelebA):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, target_type="landmarks", transform=ToTensor(), **kwargs
        )

    def __getitem__(self, idx):
        img, landmarks = super().__getitem__(idx)
        return {
            "image": img.float(),
            LEFT_EYE: landmarks[:2].float(),
            RIGHT_EYE: landmarks[2:4].float(),
            NOSE: landmarks[4:6].float(),
            LEFT_MOUTH: landmarks[6:8].float(),
            RIGHT_MOUTH: landmarks[8:].float(),
        }


# Utility to instantiate the mutual information estimators
def instantiate_estimator() -> MIEstimator:
    # Define the image encoders
    encoders = {
        RANDOM: freeze(ResNet18Encoder(weights=None)),
        PRETRAINED: freeze(ResNet18Encoder(weights="DEFAULT")),
        FINETUNED: ResNet18Encoder(weights="DEFAULT"),
    }

    # Create one q(y) for each landmark as a spline_autoregressive flow applied to a normal
    marginals = {
        landmark: transformed_normal(
            input_dim=2,
            hidden_dims=[128, 64],
            transform_name="spline_autoregressive",
            normalization="batchnorm",
            n_transforms=2,
        )
        for landmark in LANDMARKS
    }

    estimators = {}
    # We define 5x3 estimators (one for each feature and each representation) using doe (Difference of Entropies)
    for landmark in LANDMARKS:
        for representation in REPRESENTATIONS:
            # We define a proposal q(y|x) as a conditional (on x) linearly transformed Normal.
            # Note that x refers to the image representations, while y to the coordinates of each landmark.
            conditional = conditional_transformed_normal(
                input_dim=2,
                context_dim=Z_DIM,
                transform_name="conditional_linear",
                normalization="batchnorm",
                hidden_dims=[128, 64],
                n_transforms=1,
            )

            # Each generative DoE estimator relies on the estimation of the difference between H(y|x) and H(y).
            generative_estimator = DoE(
                q_Y_given_X=conditional, q_Y=marginals[landmark]
            )

            # Instead of using directly the DoE estimator, we use it in combination with NWJ to enhance the flexibility
            # of the posterior.
            estimators[
                (representation, landmark)
            ] = ResampledHybridMIEstimator(
                generative_estimator=generative_estimator,
                discriminative_estimator=nwj(
                    x_dim=Z_DIM, y_dim=2, hidden_dims=[128, 64], neg_samples=16
                ),
            )

    multi_estimator = MultiMIEstimator(estimators)

    # Since we are interested in estimating the mutual information between the resnet representations,
    mi_estimator = TransformedMIEstimator(
        # we first encode each image using the pre-trained encoder.
        transforms_rename={
            ("image", representation_name): encoder
            for representation_name, encoder in encoders.items()
        },
        # before using the specified estimator.
        base_estimator=multi_estimator,
    )

    return mi_estimator


if __name__ == "__main__":
    # Directory for the CelebA Dataset
    DATA_DIR = "/data"

    # Enable wandb logging
    wandb_logging = True

    # Device used for training the estimators
    device = "cuda"
    num_workers = 16

    # Training Parameters
    batch_size = 128
    max_epochs = 20
    optimizer_class = AdamW

    # Instantiate the dataset
    print("Loading the dataset")
    dataset = CelebALandmarks(DATA_DIR)

    train_size = int(len(dataset) * 0.9)
    train_set, valid_set = random_split(
        dataset, [train_size, len(dataset) - train_size]
    )

    # Define a simple dataloader
    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, num_workers=num_workers
    )

    # Instantiate the logger
    if wandb_logging:
        from torch_mist.utils.logging.logger.wandb import WandbLogger

        logger = WandbLogger("torch_mist")
    else:
        logger = PandasLogger()

    # Instantiate 5x3 estimators (one for each landmark and representation pair)
    # They are all wrapped into one model which takes care of encoding images and computing mutual information between
    # all pairs
    print("Instantiating the estimators")
    mi_estimator = instantiate_estimator()
    print(mi_estimator)

    # We log the value of mutual information each estimator over time
    methods_to_log = [
        f"base_estimator.estimators.{name}.log_ratio"
        for name in mi_estimator.base_estimator.estimators
    ]
    # And the overall loss
    methods_to_log += ["loss"]

    # Train the estimator
    print("Training the estimators")
    with logger.logged_methods(instance=mi_estimator, methods=methods_to_log):
        log = train_mi_estimator(
            estimator=mi_estimator,
            train_loader=train_loader,
            valid_loader=valid_loader,
            max_epochs=max_epochs,
            logger=logger,
            device=device,
            optimizer_class=optimizer_class,
        )

    print("Saving the trained models")
    extra_params = {}
    if wandb_logging:
        extra_params["artifact_name"] = "celeba_landmarks"

    logger.save_model(mi_estimator, "trained.pyt", **extra_params)
