import numpy as np
from typing import Callable


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def get_activation(fn: Callable[[np.ndarray], np.ndarray] | None) -> Callable[[np.ndarray], np.ndarray]:
    return fn if fn is not None else relu

