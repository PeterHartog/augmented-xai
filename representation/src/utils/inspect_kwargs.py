import inspect
from dataclasses import is_dataclass


def inspect_kwargs(**kwargs):
    """Wrapper for inspecting which kwargs are correct for the dataclass then inits that dataclass."""

    def decorator(cls):
        assert is_dataclass(cls), f"{cls.__name__} is not a dataclass"
        return cls(**{k: v for k, v in kwargs.items() if k in inspect.signature(cls).parameters})

    return decorator


def set_kwargs(cls, **kwargs):
    """Wrapper for inspecting which kwargs are correct for the dataclass then inits that dataclass."""
    assert is_dataclass(cls), f"{cls.__name__} is not a dataclass"
    return cls(**{k: v for k, v in kwargs.items() if k in inspect.signature(cls).parameters})
