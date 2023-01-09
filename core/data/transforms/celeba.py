# from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform
from torchvision import transforms


# Transform based on SimCLRTrainTransform in pl_bolts.models.self_supervised.simclr.transforms
class CelebATrainTransform(transforms.Compose):
    def __init__(
            self,
            image_size: int = 128,
            min_scale: float = 0.08,
            max_scale: float = 1.0,
            jitter_strength = 0.5,
            gaussian_blur: bool = False,
    ):

        t = [
            transforms.RandomResizedCrop(size=(image_size, image_size), scale=(min_scale, max_scale)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                p=0.8,
                transforms=[transforms.ColorJitter(
                    0.8 * jitter_strength,
                    0.8 * jitter_strength,
                    0.8 * jitter_strength,
                    0.2 * jitter_strength,
                    )]
            ),
            transforms.RandomGrayscale(p=0.2),
        ]
        if gaussian_blur:
            kernel_size = int(0.1 * image_size)
            if kernel_size % 2 == 0:
                kernel_size += 1

            t.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size)], p=0.5))
        t.append(transforms.ToTensor())
        super().__init__(t)


class CelebAEvalTransform(transforms.Compose):
    def __init__(
            self,
            image_size: int = 128,
    ):

        t = [
            transforms.CenterCrop(size=(176, 176)),
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor()
        ]
        super().__init__(t)

# class CelebAEvalTransform(SimCLREvalDataTransform):
#     def __init__(
#             self,
#             input_height: int = 224,
#             gaussian_blur: bool = True,
#             jitter_strength: float = 1.0,
#             normalize=None,
#             min_scale=0.6,
#             max_scale=1.0
#     ) -> None:
#         super().__init__(
#             normalize=normalize, input_height=input_height, gaussian_blur=gaussian_blur, jitter_strength=jitter_strength
#         )
#         self.train_transform.transforms[0].scale = (min_scale, max_scale)
#         self.online_transform.transforms[0].scale = (min_scale, max_scale)
