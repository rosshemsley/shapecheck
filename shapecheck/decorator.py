from typing import Any, Tuple, Union
import inspect

from .enabled import __CHECKS_ENABLED


def check_args(*label_args, **label_kwargs):
    if len(label_args) != 0:
        raise ValueError("all arguments must be named arguments")

    def decorator(func):
        if not __CHECKS_ENABLED:
            return func
        argspec = inspect.getfullargspec(func)

        def wrapper(*args, **kwargs):
            for k, v in label_kwargs.items():
                idx = argspec.args.index(k)
                try:
                    check(args[idx], v)
                except IndexError:
                    check(kwargs[k], v)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def check(value: Any, label: Tuple[Union[str, Tuple[str]]]):
    match, msg = _types_match(value, label)
    if not match:
        raise TypeError(msg)


def _types_match(value, label) -> (bool, str):
    if len(label) != len(value.shape):
        return (
            False,
            f"lengths do not match: expected {label} of length {len(label)}, got {value.shape}",
        )

    for i, v in enumerate(label):
        if isinstance(v, str):
            continue
        elif isinstance(v, int):
            if v != value.shape[i]:
                return (
                    False,
                    f"got tensor with shape {value.shape}. Dimension {i} should match pattern {v} of size {len(v)}, but got size {value.shape[i]} instead.",
                )
        elif isinstance(v, (list, tuple)):
            if len(v) != value.shape[i]:
                return (
                    False,
                    f"got tensor with shape {value.shape}. Dimension {i} should match pattern {v} of size {len(v)}, but got size {value.shape[i]} instead.",
                )

    return True, ""


def _validate_target(target):
    if not isinstance(target, (tuple, list)):
        raise ValueError("target must be tuple or list")

    for v in target:
        if isinstance(v, str):
            continue
        elif isinstance(v, tuple):
            if not all(isinstance(s, str) for s in v):
                raise ValueError("nested tuples must contain only strings")
        else:
            raise ValueError("target must contain only tuples or strings")
