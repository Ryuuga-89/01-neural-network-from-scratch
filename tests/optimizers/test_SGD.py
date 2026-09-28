import numpy as np
import torch

from src.neural_networks.layers.basic_layers import Linear, ReLU
from src.neural_networks.layers.other_layers import Sequential
from src.neural_networks.optimizers.optimizers import SGD

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
    
    
def test_sgd_single_linear_step():
    model = Sequential([
        Linear(shape=(2, 3), args=LINEAR_ARGS),
    ])

    linear = model.layers[0]

    x = np.array([
        [0.2, -0.3],
        [0.4, 0.1],
    ])

    dout = np.array([
        [0.1, -0.2, 0.3],
        [-0.4, 0.5, 0.6],
    ])

    # gradientを生成
    model.forward_propagation(x)
    model.backward_propagation(dout)
    
    assert isinstance(linear, Linear)

    old_W = linear.W.copy()
    old_b = linear.b.copy()

    dW = linear.dW.copy()
    db = linear.db.copy()

    lr = 0.01

    expected_W = old_W - lr * dW
    expected_b = old_b - lr * db

    optimizer = SGD(model, lr=lr)
    optimizer.step()

    np.testing.assert_allclose(
        linear.W,
        expected_W,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear.b,
        expected_b,
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_sgd_against_pytorch():
    model = create_model()

    linear1 = model.layers[0]
    linear2 = model.layers[2]

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

    lr = 0.01
    
    assert isinstance(linear1, Linear)
    assert isinstance(linear2, Linear)

    # 更新前parameterを保存
    old_W1 = linear1.W.copy()
    old_b1 = linear1.b.copy()
    old_W2 = linear2.W.copy()
    old_b2 = linear2.b.copy()

    # =========================
    # 自作NN
    # =========================

    model.forward_propagation(x)
    model.backward_propagation(dout)

    optimizer = SGD(model, lr=lr)
    optimizer.step()

    # =========================
    # PyTorch
    # =========================

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
    )

    W1 = torch.tensor(
        old_W1,
        dtype=torch.float64,
        requires_grad=True,
    )

    b1 = torch.tensor(
        old_b1,
        dtype=torch.float64,
        requires_grad=True,
    )

    W2 = torch.tensor(
        old_W2,
        dtype=torch.float64,
        requires_grad=True,
    )

    b2 = torch.tensor(
        old_b2,
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

    torch_optimizer = torch.optim.SGD(
        [W1, b1, W2, b2],
        lr=lr,
    )

    torch_optimizer.step()

    # =========================
    # 比較
    # =========================

    np.testing.assert_allclose(
        linear1.W,
        W1.detach().numpy(),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear1.b,
        b1.detach().numpy(),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear2.W,
        W2.detach().numpy(),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear2.b,
        b2.detach().numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_sgd_zero_learning_rate():
    model = create_model()

    linear1 = model.layers[0]
    linear2 = model.layers[2]

    x = np.array([
        [0.2, -0.3],
        [0.4, 0.1],
    ])

    dout = np.array([
        [0.1, -0.2],
        [0.3, 0.4],
    ])

    model.forward_propagation(x)
    model.backward_propagation(dout)
    
    assert isinstance(linear1, Linear)
    assert isinstance(linear2, Linear)

    old_W1 = linear1.W.copy()
    old_b1 = linear1.b.copy()
    old_W2 = linear2.W.copy()
    old_b2 = linear2.b.copy()

    optimizer = SGD(model, lr=0.0)
    optimizer.step()

    np.testing.assert_array_equal(
        linear1.W,
        old_W1,
    )

    np.testing.assert_array_equal(
        linear1.b,
        old_b1,
    )

    np.testing.assert_array_equal(
        linear2.W,
        old_W2,
    )

    np.testing.assert_array_equal(
        linear2.b,
        old_b2,
    )