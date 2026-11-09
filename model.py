"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
import numpy as np 

def numerical_gradient(f, x, eps=1e-5):
    x = np.array(x, dtype=float, copy=True)
    grad = np.zeros_like(x)

    for idx in np.ndindex(x.shape):
        original = x[idx]

        x[idx] = original + eps
        f_plus = f(x)

        x[idx] = original - eps
        f_minus = f(x)

        x[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    
    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    a = np.asarray(analytic_grad, dtype=float)
    n = np.asarray(numeric_grad, dtype=float)

    denominator = np.maximum(tol, np.maximum(np.abs(a), np.abs(n)))
    relative_error = np.abs(a - n) / denominator

    return float(np.max(relative_error, initial=0.0))

# Step 3 - make_dense
import numpy as np

def make_dense(in_dim, out_dim, weight_init_fn):
    W, b = weight_init_fn(in_dim, out_dim)
    params = {"W": W, "b": b}

    def forward(x):
        y = x @ params["W"] + params["b"]
        cache = (x, params["W"].copy())
        return y, cache

    def backward(dout, cache):
        x, W = cache

        dx = dout @ W.T
        grads = {
            "W": x.T @ dout,
            "b": np.sum(dout, axis=0),
        }
        return dx, grads

    return {
        "params": params,
        "forward": forward,
        "backward": backward,
    }

# Step 4 - make_activation
import numpy as np

def make_activation(kind="relu"):
    if kind != "relu":
        raise ValueError(f"Unsupported activation: {kind}")

    def forward(x):
        y = np.maximum(0, x)
        cache = x > 0
        return y, cache

    def backward(dout, cache):
        dx = dout * cache
        return dx, {}

    return {
        "params": {},
        "forward": forward,
        "backward": backward,
    }

# Step 5 - initialize_weights
import numpy as np

def initialize_weights(in_dim, out_dim, scheme="he"):
    if scheme == "he":
        scale = np.sqrt(2.0 / in_dim)
    elif scheme == "xavier":
        scale = np.sqrt(2.0 / (in_dim + out_dim))
    else:
        raise ValueError(f"Unsupported initialization: {scheme}")

    W = np.random.randn(in_dim, out_dim) * scale
    b = np.zeros(out_dim)

    return W, b

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

