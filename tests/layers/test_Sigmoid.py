import numpy as np
import torch

from src.neural_networks.layers.basic_layers import Sigmoid


def test_sigmoid_forward():
    sigmoid = Sigmoid()

    x = np.array([
        [-2.0, -1.0, 0.0],
        [1.0, 2.0, 3.0],
    ])

    actual = sigmoid.forward_propagation(x)

    expected = 1.0 / (1.0 + np.exp(-x))

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )


def test_sigmoid_forward_known_values():
    sigmoid = Sigmoid()

    x = np.array([
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
    ])

    actual = sigmoid.forward_propagation(x)

    expected = np.array([
        0.11920292202211755,
        0.2689414213699951,
        0.5,
        0.7310585786300049,
        0.8807970779778823,
    ])

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )


def test_sigmoid_forward_against_pytorch():
    sigmoid = Sigmoid()

    x = np.array([
        [-2.0, -0.5, 0.0],
        [0.5, 1.0, 3.0],
    ])

    actual = sigmoid.forward_propagation(x)

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
    )

    expected = torch.sigmoid(x_torch)

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )


def test_sigmoid_backward():
    sigmoid = Sigmoid()

    x = np.array([
        [-2.0, -0.5, 0.0],
        [0.5, 1.0, 2.0],
    ])

    dout = np.array([
        [0.1, -0.2, 0.3],
        [0.4, -0.5, 0.6],
    ])

    out = sigmoid.forward_propagation(x)
    actual = sigmoid.backward_propagation(dout)

    expected = dout * out * (1.0 - out)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )


def test_sigmoid_backward_against_pytorch():
    sigmoid = Sigmoid()

    x = np.array([
        [-2.0, -0.5, 0.0],
        [0.5, 1.0, 2.0],
    ])

    dout = np.array([
        [0.1, -0.2, 0.3],
        [0.4, -0.5, 0.6],
    ])

    # 自作実装
    sigmoid.forward_propagation(x)
    actual = sigmoid.backward_propagation(dout)

    # PyTorch
    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
        requires_grad=True,
    )

    dout_torch = torch.tensor(
        dout,
        dtype=torch.float64,
    )

    y_torch = torch.sigmoid(x_torch)
    y_torch.backward(dout_torch)

    assert x_torch.grad is not None

    expected = x_torch.grad.numpy()

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )


def test_sigmoid_backward_numerical_gradient():
    sigmoid = Sigmoid()

    x = np.array([
        [-1.5, -0.3],
        [0.7, 2.0],
    ])

    dout = np.array([
        [0.2, -0.4],
        [0.7, 0.3],
    ])

    # analytical gradient
    sigmoid.forward_propagation(x)
    analytical = sigmoid.backward_propagation(dout)

    # numerical gradient
    eps = 1e-5
    numerical = np.zeros_like(x)

    for index in np.ndindex(x.shape):
        original = x[index]

        x[index] = original + eps
        sigmoid_plus = Sigmoid()
        out_plus = sigmoid_plus.forward_propagation(x)
        loss_plus = np.sum(out_plus * dout)

        x[index] = original - eps
        sigmoid_minus = Sigmoid()
        out_minus = sigmoid_minus.forward_propagation(x)
        loss_minus = np.sum(out_minus * dout)

        numerical[index] = (
            loss_plus - loss_minus
        ) / (2 * eps)

        x[index] = original

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=1e-5,
        atol=1e-6,
    )


def test_sigmoid_zero():
    sigmoid = Sigmoid()

    x = np.array([0.0])

    actual = sigmoid.forward_propagation(x)

    np.testing.assert_allclose(
        actual,
        np.array([0.5]),
        rtol=1e-7,
        atol=1e-7,
    )


def test_sigmoid_output_range():
    sigmoid = Sigmoid()

    x = np.linspace(
        -10.0,
        10.0,
        100,
    )

    actual = sigmoid.forward_propagation(x)

    assert np.all(actual > 0.0)
    assert np.all(actual < 1.0)