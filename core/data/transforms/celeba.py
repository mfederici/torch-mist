from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform


class CelebATrainTransform(SimCLRTrainDataTransform):
    def __init__(
            self,
            input_height: int = 224,
            gaussian_blur: bool = True,
            jitter_strength: float = 1.0,
            normalize=None,
            min_scale=0.6,
            max_scale=1.0
    ) -> None:
        super().__init__(
            normalize=normalize, input_height=input_height, gaussian_blur=gaussian_blur, jitter_strength=jitter_strength
        )
        self.train_transform.transforms[0].scale = (min_scale, max_scale)
        self.online_transform.transforms[0].scale = (min_scale, max_scale)


class CelebAEvalTransform(SimCLREvalDataTransform):
    def __init__(
            self,
            input_height: int = 224,
            gaussian_blur: bool = True,
            jitter_strength: float = 1.0,
            normalize=None,
            min_scale=0.6,
            max_scale=1.0
    ) -> None:
        super().__init__(
            normalize=normalize, input_height=input_height, gaussian_blur=gaussian_blur, jitter_strength=jitter_strength
        )
        self.train_transform.transforms[0].scale = (min_scale, max_scale)
        self.online_transform.transforms[0].scale = (min_scale, max_scale)
