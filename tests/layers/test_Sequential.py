import numpy as np
import torch

from src.neural_networks.layers.abstract_layers import Layer, ParametricLayer
from src.neural_networks.layers.basic_layers import Linear, ReLU
from src.neural_networks.layers.other_layers import Sequential


LINEAR_ARGS = {
    "weight_init_method": {
        "method_name": "He",
        "distribution": "normal",
        "seed": 42,
    },
    "bias_init_method": {
        "method_name": "Zeros",
    },
}


def create_model() -> Sequential:
    return Sequential([
        Linear(shape=(2, 3), args=LINEAR_ARGS),
        ReLU(),
        Linear(shape=(3, 2), args=LINEAR_ARGS),
    ])
    

def test_sequential_forward_against_pytorch():
    model = create_model()

    x = np.array([
        [0.2, -0.3],
        [0.4, 0.1],
        [-0.5, 0.8],
    ])

    actual = model.forward_propagation(x)

    linear1 = model.layers[0]
    linear2 = model.layers[2]

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
    )
    
    assert isinstance(linear1, Linear)
    assert isinstance(linear2, Linear)

    W1 = torch.tensor(
        linear1.W,
        dtype=torch.float64,
    )
    b1 = torch.tensor(
        linear1.b,
        dtype=torch.float64,
    )

    W2 = torch.tensor(
        linear2.W,
        dtype=torch.float64,
    )
    b2 = torch.tensor(
        linear2.b,
        dtype=torch.float64,
    )

    expected = x_torch @ W1 + b1
    expected = torch.relu(expected)
    expected = expected @ W2 + b2

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_sequential_backward_against_pytorch():
    model = create_model()

    x = np.array([
        [0.2, -0.3],
        [0.4, 0.1],
        [-0.5, 0.8],
    ])

    dout = np.array([
        [0.1, -0.2],
        [0.3, 0.4],
        [-0.5, 0.6],
    ])

    # -------------------------
    # 自作NN
    # -------------------------

    model.forward_propagation(x)
    actual_dx = model.backward_propagation(dout)

    linear1 = model.layers[0]
    linear2 = model.layers[2]

    # -------------------------
    # PyTorch
    # -------------------------

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
        requires_grad=True,
    )
    
    assert isinstance(linear1, Linear)
    assert isinstance(linear2, Linear)

    W1 = torch.tensor(
        linear1.W,
        dtype=torch.float64,
        requires_grad=True,
    )
    b1 = torch.tensor(
        linear1.b,
        dtype=torch.float64,
        requires_grad=True,
    )

    W2 = torch.tensor(
        linear2.W,
        dtype=torch.float64,
        requires_grad=True,
    )
    b2 = torch.tensor(
        linear2.b,
        dtype=torch.float64,
        requires_grad=True,
    )

    y = x_torch @ W1 + b1
    y = torch.relu(y)
    y = y @ W2 + b2

    dout_torch = torch.tensor(
        dout,
        dtype=torch.float64,
    )

    y.backward(dout_torch)

    # -------------------------
    # 比較
    # -------------------------

    assert x_torch.grad is not None
    assert W1.grad is not None
    assert b1.grad is not None
    assert W2.grad is not None
    assert b2.grad is not None

    np.testing.assert_allclose(
        actual_dx,
        x_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear1.dW,
        W1.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear1.db,
        b1.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear2.dW,
        W2.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear2.db,
        b2.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_sequential_train_mode():
    model = create_model()

    assert model.train is True

    model.train = False

    assert model.train is False

    for layer in model.layers:
        assert layer.train is False

    model.train = True

    assert model.train is True

    for layer in model.layers:
        assert layer.train is True
        
        
def get_params(self) -> list[tuple[object, str, str]]:
    params = []

    for layer in self.layers:
        params.extend(layer.get_params())

    return params


def test_sequential_get_params():
    model = create_model()

    params = model.get_params()

    assert len(params) == 4

    linear1 = model.layers[0]
    linear2 = model.layers[2]

    assert params[0] == (linear1, "W", "dW")
    assert params[1] == (linear1, "b", "db")
    assert params[2] == (linear2, "W", "dW")
    assert params[3] == (linear2, "b", "db")
    
    
