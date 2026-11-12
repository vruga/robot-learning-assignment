"""JSON + base64 ndarray transport. Never unpickle candidate-provided bytes."""

import base64

import numpy as np


def pack_array(array):
    array = np.ascontiguousarray(array)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def unpack_array(value):
    dtype = np.dtype(value["dtype"])
    if dtype not in (np.dtype("uint8"), np.dtype("float32")):
        raise ValueError("Unsupported wire dtype")
    return (
        np.frombuffer(base64.b64decode(value["data"], validate=True), dtype=dtype)
        .reshape(value["shape"])
        .copy()
    )
