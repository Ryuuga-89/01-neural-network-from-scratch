import numpy as np
import torch

from src.neural_networks.layers.basic_layers import ReLU


def test_relu_forward():
    relu = ReLU()

    x = np.array([
        [-2.0, -1.0, 0.5],
        [1.0, 2.0, -3.0],
    ])

    expected = np.array([
        [0.0, 0.0, 0.5],
        [1.0, 2.0, 0.0],
    ])

    actual = relu.forward_propagation(x)

    np.testing.assert_array_equal(actual, expected)


def test_relu_backward():
    relu = ReLU()

    x = np.array([
        [-2.0, -1.0, 0.5],
        [1.0, 2.0, -3.0],
    ])

    dout = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    relu.forward_propagation(x)
    dx = relu.backward_propagation(dout)

    expected = np.array([
        [0.0, 0.0, 3.0],
        [4.0, 5.0, 0.0],
    ])

    np.testing.assert_array_equal(dx, expected)
    

def test_relu_forward_against_pytorch():
    relu = ReLU()

    x = np.array([
        [-2.3, -0.7, 0.2],
        [1.4, 3.1, -4.2],
    ])

    actual = relu.forward_propagation(x)

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
    )

    expected = torch.relu(x_torch)

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )


def test_relu_backward_against_pytorch():
    relu = ReLU()

    x = np.array([
        [-2.3, -0.7, 0.2],
        [1.4, 3.1, -4.2],
    ])

    dout = np.array([
        [0.1, -0.2, 0.3],
        [-0.4, 0.5, 0.6],
    ])

    # 自作
    relu.forward_propagation(x)
    dx = relu.backward_propagation(dout)

    # PyTorch
    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
        requires_grad=True,
    )

    y_torch = torch.relu(x_torch)

    dout_torch = torch.tensor(
        dout,
        dtype=torch.float64,
    )

    y_torch.backward(dout_torch)

    assert x_torch.grad is not None
    np.testing.assert_allclose(
        dx,
        x_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_relu_at_zero():
    relu = ReLU()

    x = np.array([0.0])

    relu.forward_propagation(x)

    dx = relu.backward_propagation(
        np.array([1.0])
    )

    np.testing.assert_array_equal(
        dx,
        np.array([0.0]),
    )