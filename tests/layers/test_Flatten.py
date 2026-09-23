import numpy as np
import torch

from src.neural_networks.layers.basic_layers import Flatten


def test_flatten_forward():
    flatten = Flatten()

    x = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    actual = flatten.forward_propagation(x)

    expected = x.reshape(2, -1)

    assert actual.shape == (2, 12)

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_flatten_forward_preserves_batch_dimension():
    flatten = Flatten()

    x = np.arange(
        5 * 3 * 4 * 2
    ).reshape(5, 3, 4, 2)

    actual = flatten.forward_propagation(x)

    assert actual.shape == (5, 24)


def test_flatten_forward_preserves_values():
    flatten = Flatten()

    x = np.array([
        [
            [1, 2],
            [3, 4],
        ],
        [
            [5, 6],
            [7, 8],
        ],
    ])

    actual = flatten.forward_propagation(x)

    expected = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ])

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_flatten_backward():
    flatten = Flatten()

    x = np.arange(
        2 * 3 * 4
    ).reshape(2, 3, 4)

    flatten.forward_propagation(x)

    dout = np.arange(
        2 * 12
    ).reshape(2, 12)

    actual = flatten.backward_propagation(dout)

    expected = dout.reshape(2, 3, 4)

    assert actual.shape == x.shape

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_flatten_forward_against_pytorch():
    flatten = Flatten()

    x = np.arange(
        2 * 3 * 4 * 5,
        dtype=np.float64,
    ).reshape(2, 3, 4, 5)

    actual = flatten.forward_propagation(x)

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
    )

    expected = torch.flatten(
        x_torch,
        start_dim=1,
    )

    np.testing.assert_array_equal(
        actual,
        expected.numpy(),
    )


def test_flatten_backward_against_pytorch():
    flatten = Flatten()

    x = np.arange(
        2 * 3 * 4,
        dtype=np.float64,
    ).reshape(2, 3, 4)

    dout = np.array([
        [
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2,
        ],
        [
            -0.1, -0.2, -0.3, -0.4,
            -0.5, -0.6, -0.7, -0.8,
            -0.9, -1.0, -1.1, -1.2,
        ],
    ])

    # 自作実装
    flatten.forward_propagation(x)
    actual = flatten.backward_propagation(dout)

    # PyTorch
    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
        requires_grad=True,
    )

    y_torch = torch.flatten(
        x_torch,
        start_dim=1,
    )

    dout_torch = torch.tensor(
        dout,
        dtype=torch.float64,
    )

    y_torch.backward(dout_torch)

    assert x_torch.grad is not None

    expected = x_torch.grad.numpy()

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_flatten_2d_input():
    """
    すでに(batch, features)の場合は
    shapeも値も変化しないことを確認する。
    """
    flatten = Flatten()

    x = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    actual = flatten.forward_propagation(x)

    assert actual.shape == (2, 3)

    np.testing.assert_array_equal(
        actual,
        x,
    )


def test_flatten_single_sample_batch():
    flatten = Flatten()

    x = np.arange(
        3 * 4 * 5
    ).reshape(1, 3, 4, 5)

    actual = flatten.forward_propagation(x)

    assert actual.shape == (1, 60)

    restored = flatten.backward_propagation(actual)

    assert restored.shape == (1, 3, 4, 5)

    np.testing.assert_array_equal(
        restored,
        x,
    )