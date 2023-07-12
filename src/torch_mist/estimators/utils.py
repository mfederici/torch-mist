import inspect


def instantiate_estimator(estimator_name, verbose=True, **kwargs):
    import torch_mist.estimators as estimators
    if not hasattr(estimators, estimator_name):
        raise Exception(f"Estimator {estimator_name} not found")
    factory = getattr(estimators, estimator_name)
    new_kwargs = {}
    unused_kwargs = []
    for name, value in kwargs.items():
        if name in inspect.signature(factory).parameters:
            new_kwargs[name] = value
        else:
            unused_kwargs.append(name)
    if len(unused_kwargs) > 0 and verbose:
        print(f"Warning: parameter(s) {unused_kwargs} are not used for {estimator_name}")
    return factory(**new_kwargs)


