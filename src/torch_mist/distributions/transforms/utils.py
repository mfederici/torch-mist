def fetch_transform(transform_name: str):
    import pyro.distributions.transforms as pyro_transforms_module
    import torch_mist.distributions.transforms.factories as transforms_module

    if hasattr(pyro_transforms_module, transform_name):
        transform_factory = getattr(pyro_transforms_module, transform_name)
    elif hasattr(transforms_module, transform_name):
        transform_factory = getattr(transforms_module, transform_name)
    else:
        raise NotImplementedError(
            f"Transform {transform_name} is not implemented."
        )
    return transform_factory
