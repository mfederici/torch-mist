import inspect


def args_to_kwargs(method, args, kwargs):
    keys = inspect.signature(method).parameters.keys()
    unused_keys = [key for key in keys if key not in kwargs]
    if "self" in unused_keys:
        unused_keys.remove("self")
    new_kwargs = {unused_keys[i]: arg for i, arg in enumerate(args)}
    kwargs = {**kwargs, **new_kwargs}
    return kwargs
