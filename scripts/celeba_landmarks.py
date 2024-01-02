from copy import deepcopy

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
from torch_mist.utils import train_mi_estimator
from torch_mist.estimators import TransformedMIEstimator, DoE
from torch_mist.estimators.multi import MultiMIEstimator

# Dimensionality of the resnet18 representation
Z_DIM = 512

# Intermediate layers of the mutual information estimators
HIDDEN_DIMS = [64, 64]


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


# We define a simple utility to normalize the position of the face landmarks
def normalize(coord):
    coord = coord - torch.FloatTensor([218, 178]) / 2.0
    coord = coord / torch.FloatTensor([218, 178])
    return coord


# utility to map normalized coordinates into the original space
def revert(coord):
    coord = coord * torch.FloatTensor([218, 178])
    coord += torch.FloatTensor([218, 178]) / 2.0
    return coord


LEFT_EYE = "left_eye"
RIGHT_EYE = "right_eye"
NOSE = "nose"
RIGHT_MOUTH = "right_mouth"
LEFT_MOUTH = "left_mouth"
LANDMARKS = [LEFT_EYE, RIGHT_EYE, NOSE, LEFT_MOUTH, RIGHT_MOUTH]

RANDOM = "random_init"
PRETRAINED = "pretrained"
FINETUNED = "finetuned"
REPRESENTATIONS = [RANDOM, PRETRAINED, FINETUNED]


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
            LEFT_EYE: normalize(landmarks[:2].float()),
            RIGHT_EYE: normalize(landmarks[2:4].float()),
            NOSE: normalize(landmarks[4:6].float()),
            LEFT_MOUTH: normalize(landmarks[6:8].float()),
            RIGHT_MOUTH: normalize(landmarks[8:].float()),
        }


if __name__ == "__main__":
    from torch_mist.utils.logging.logger.wandb import WandbLogger

    landmarks = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]
    encoders = {
        "pretrained": freeze(ResNet18Encoder(weights="DEFAULT")),
        "finetuned": ResNet18Encoder(weights="DEFAULT"),
        "random_init": freeze(ResNet18Encoder(weights=None)),
    }
    dataset = CelebALandmarks("/data")

    # Lastly, we define a dataloader
    dataloader = DataLoader(
        dataset, batch_size=128, num_workers=8, shuffle=True
    )

    # Device used for training the estimators
    device = "cuda"

    marginals = {
        landmark: transformed_normal(
            input_dim=2,
            hidden_dims=HIDDEN_DIMS,
            transform_name="spline_autoregressive",
            n_transforms=2,
        )
        for landmark in landmarks
    }

    conditional = conditional_transformed_normal(
        input_dim=2,
        context_dim=Z_DIM,
        transform_name="conditional_spline_autoregressive",
        hidden_dims=HIDDEN_DIMS,
        n_transforms=2,
    )

    estimators = {}
    # We define 5x2 estimators (one for each feature and each representation) using doe (Difference of Entropies)
    for landmark in landmarks:
        for representation in encoders:
            # Each estimator relies on the estimation of the difference between H(y|x) and H(y).
            # Both entropies are estimated using variational distribution transformed using spline_autoregressive flows.
            # Note that x refers to the image representations, while y to the coordinates of each landmark.
            estimators[(representation, landmark)] = DoE(
                q_Y_given_X=deepcopy(conditional), q_Y=marginals[landmark]
            )

    multi_estimator = MultiMIEstimator(estimators)

    # Since we are interested in estimating the mutual information between the resnet representations,
    mi_estimator = TransformedMIEstimator(
        transforms={},
        # we first encode each image using the pre-trained encoder.
        transforms_rename={
            ("image", representation_name): encoder
            for representation_name, encoder in encoders.items()
        },
        # before using the specified estimator.
        base_estimator=multi_estimator,
    )

    logger = WandbLogger("torch_mist")
    methods_to_log = [
        f"{name}.mutual_information"
        for name in mi_estimator.base_estimator.estimators
    ] + [f"{name}.loss" for name in mi_estimator.base_estimator.estimators]

    with logger.logged_methods(
        instance=mi_estimator.base_estimator.estimators, methods=methods_to_log
    ):
        log = train_mi_estimator(
            estimator=mi_estimator,
            train_loader=dataloader,
            max_epochs=20,
            verbose=True,
            logger=logger,
            device="cuda",
        )

    logger.save_model(mi_estimator, "trained.pyt")
