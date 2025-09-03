import os
import random
import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)
        if hasattr(tf, "set_random_seed"):
            tf.set_random_seed(seed)
    try:
        import keras.backend as K  # type: ignore
        if hasattr(K, "set_random_seed"):
            K.set_random_seed(seed)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
