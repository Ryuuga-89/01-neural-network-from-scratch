import numpy as np
import pytest
import torch

from src.neural_networks.loss_functions.loss_functions import MeanSquaredErrorLoss


def test_mse_forward_known_values():
    loss_function = MeanSquaredErrorLoss()

    output = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    t = np.array([
        [0.0, 2.0],
        [5.0, 1.0],
    ])

    actual = loss_function.forward_propagation(
        output,
        t,
    )

    # squared error:
    #
    # (1 - 0)^2 = 1
    # (2 - 2)^2 = 0
    # (3 - 5)^2 = 4
    # (4 - 1)^2 = 9
    #
    # sum = 14
    # 0.5 * 14 / batch_size(2) = 3.5
    expected = 3.5

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )


def test_mse_backward_known_values():
    loss_function = MeanSquaredErrorLoss()

    output = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    t = np.array([
        [0.0, 2.0],
        [5.0, 1.0],
    ])

    loss_function.forward_propagation(
        output,
        t,
    )

    actual = loss_function.backward_propagation()

    # dL/doutput = (output - t) / batch_size
    expected = np.array([
        [0.5, 0.0],
        [-1.0, 1.5],
    ])

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_mse_forward_against_pytorch():
    rng = np.random.default_rng(123)

    output = rng.normal(
        size=(5, 3)
    )

    t = rng.normal(
        size=(5, 3)
    )

    loss_function = MeanSquaredErrorLoss()

    actual = loss_function.forward_propagation(
        output,
        t,
    )

    output_torch = torch.tensor(
        output,
        dtype=torch.float64,
    )

    t_torch = torch.tensor(
        t,
        dtype=torch.float64,
    )

    batch_size = output.shape[0]

    expected = (
        0.5
        * torch.nn.functional.mse_loss(
            output_torch,
            t_torch,
            reduction="sum",
        )
        / batch_size
    )

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_mse_backward_against_pytorch():
    rng = np.random.default_rng(123)

    output = rng.normal(
        size=(5, 3)
    )

    t = rng.normal(
        size=(5, 3)
    )

    loss_function = MeanSquaredErrorLoss()

    loss_function.forward_propagation(
        output,
        t,
    )

    actual = loss_function.backward_propagation()

    output_torch = torch.tensor(
        output,
        dtype=torch.float64,
        requires_grad=True,
    )

    t_torch = torch.tensor(
        t,
        dtype=torch.float64,
    )

    batch_size = output.shape[0]

    loss_torch = (
        0.5
        * torch.nn.functional.mse_loss(
            output_torch,
            t_torch,
            reduction="sum",
        )
        / batch_size
    )

    loss_torch.backward()

    assert output_torch.grad is not None

    expected = output_torch.grad.numpy()

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_mse_backward_with_dout():
    loss_function = MeanSquaredErrorLoss()

    output = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    t = np.array([
        [0.0, 2.0],
        [5.0, 1.0],
    ])

    loss_function.forward_propagation(
        output,
        t,
    )

    dout = 2.5

    actual = loss_function.backward_propagation(
        dout=dout,
    )

    expected = (
        dout
        * (output - t)
        / output.shape[0]
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_mse_backward_numerical_gradient():
    rng = np.random.default_rng(123)

    output = rng.normal(
        size=(3, 2)
    )

    t = rng.normal(
        size=(3, 2)
    )

    loss_function = MeanSquaredErrorLoss()

    loss_function.forward_propagation(
        output,
        t,
    )

    analytical = (
        loss_function
        .backward_propagation()
    )

    eps = 1e-5
    numerical = np.zeros_like(output)

    for index in np.ndindex(output.shape):
        original = output[index]

        output[index] = original + eps

        loss_plus = MeanSquaredErrorLoss()
        l_plus = loss_plus.forward_propagation(
            output,
            t,
        )

        output[index] = original - eps

        loss_minus = MeanSquaredErrorLoss()
        l_minus = loss_minus.forward_propagation(
            output,
            t,
        )

        numerical[index] = (
            l_plus - l_minus
        ) / (2 * eps)

        output[index] = original

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=1e-5,
        atol=1e-6,
    )
    
    
def test_mse_rejects_different_shapes():
    loss_function = MeanSquaredErrorLoss()

    output = np.zeros((4, 3))
    t = np.zeros((4, 2))

    with pytest.raises(
        ValueError,
        match="予測値と期待値の形状が異なります",
    ):
        loss_function.forward_propagation(
            output,
            t,
        )
        
        
def test_mse_zero_when_prediction_equals_target():
    loss_function = MeanSquaredErrorLoss()

    output = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    t = output.copy()

    actual_loss = (
        loss_function.forward_propagation(
            output,
            t,
        )
    )

    actual_gradient = (
        loss_function.backward_propagation()
    )

    assert actual_loss == 0.0

    np.testing.assert_array_equal(
        actual_gradient,
        np.zeros_like(output),
    )
    
    
def test_mse_1d_is_treated_as_single_sample():
    loss_function = MeanSquaredErrorLoss()

    output = np.array([
        1.0,
        2.0,
        3.0,
    ])

    t = np.array([
        0.0,
        0.0,
        0.0,
    ])

    actual_loss = (
        loss_function.forward_propagation(
            output,
            t,
        )
    )

    # 0.5 * (1 + 4 + 9)
    expected_loss = 7.0

    np.testing.assert_allclose(
        actual_loss,
        expected_loss,
    )

    actual_gradient = (
        loss_function.backward_propagation()
    )

    expected_gradient = np.array([
        1.0,
        2.0,
        3.0,
    ])

    np.testing.assert_array_equal(
        actual_gradient,
        expected_gradient,
    )