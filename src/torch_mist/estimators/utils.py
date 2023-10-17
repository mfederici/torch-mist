def instantiate_estimator(estimator_name, **kwargs):
    import torch_mist.estimators as estimators

    factory = getattr(estimators, estimator_name)
    estimator = factory(**kwargs)
    return estimator
