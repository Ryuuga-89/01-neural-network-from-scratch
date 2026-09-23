import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.neural_networks.loss_functions.loss_functions import BCEWithLogitsLoss


def test_bce_with_logits_forward_known_values():
    loss_function = BCEWithLogitsLoss()

    output = np.array([
        [0.0],
        [1.0],
        [-1.0],
        [2.0],
    ])

    t = np.array([
        [0.0],
        [1.0],
        [0.0],
        [1.0],
    ])

    actual = loss_function.forward_propagation(
        output,
        t,
    )

    # 安定なBCEWithLogitsの定義
    loss_matrix = (
        np.maximum(output, 0)
        - output * t
        + np.log1p(np.exp(-np.abs(output)))
    )

    expected = np.sum(loss_matrix) / output.shape[0]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_bce_with_logits_forward_against_pytorch():
    rng = np.random.default_rng(123)

    output = rng.normal(
        size=(8, 3)
    )

    t = rng.uniform(
        0.0,
        1.0,
        size=(8, 3),
    )

    loss_function = BCEWithLogitsLoss()

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

    expected = (
        F.binary_cross_entropy_with_logits(
            output_torch,
            t_torch,
            reduction="sum",
        )
        / output.shape[0]
    )

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_bce_with_logits_backward_known_values():
    loss_function = BCEWithLogitsLoss()

    output = np.array([
        [0.0],
        [1.0],
        [-1.0],
    ])

    t = np.array([
        [0.0],
        [1.0],
        [0.0],
    ])

    loss_function.forward_propagation(
        output,
        t,
    )

    actual = loss_function.backward_propagation()

    sigmoid = 1.0 / (
        1.0 + np.exp(-output)
    )

    expected = (
        sigmoid - t
    ) / output.shape[0]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_bce_with_logits_backward_against_pytorch():
    rng = np.random.default_rng(123)

    output = rng.normal(
        size=(8, 3)
    )

    t = rng.uniform(
        0.0,
        1.0,
        size=(8, 3),
    )

    loss_function = BCEWithLogitsLoss()

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

    loss_torch = (
        F.binary_cross_entropy_with_logits(
            output_torch,
            t_torch,
            reduction="sum",
        )
        / output.shape[0]
    )

    loss_torch.backward()

    assert output_torch.grad is not None

    np.testing.assert_allclose(
        actual,
        output_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_bce_with_logits_backward_numerical_gradient():
    rng = np.random.default_rng(123)

    output = rng.normal(
        size=(4, 2)
    )

    t = rng.uniform(
        0.0,
        1.0,
        size=(4, 2),
    )

    loss_function = BCEWithLogitsLoss()

    loss_function.forward_propagation(
        output,
        t,
    )

    analytical = (
        loss_function.backward_propagation()
    )

    eps = 1e-5
    numerical = np.zeros_like(output)

    for index in np.ndindex(output.shape):
        original = output[index]

        output[index] = original + eps

        loss_plus = BCEWithLogitsLoss()
        l_plus = loss_plus.forward_propagation(
            output,
            t,
        )

        output[index] = original - eps

        loss_minus = BCEWithLogitsLoss()
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
    
    
def test_bce_with_logits_backward_with_dout():
    loss_function = BCEWithLogitsLoss()

    output = np.array([
        [0.5],
        [-1.0],
        [2.0],
    ])

    t = np.array([
        [1.0],
        [0.0],
        [1.0],
    ])

    loss_function.forward_propagation(
        output,
        t,
    )

    dout = 2.5

    actual = loss_function.backward_propagation(
        dout=dout,
    )

    sigmoid = 1.0 / (
        1.0 + np.exp(-output)
    )

    expected = (
        dout
        * (sigmoid - t)
        / output.shape[0]
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_bce_with_logits_numerical_stability():
    loss_function = BCEWithLogitsLoss()

    output = np.array([
        [1000.0],
        [-1000.0],
        [1000.0],
        [-1000.0],
    ])

    t = np.array([
        [1.0],
        [0.0],
        [0.0],
        [1.0],
    ])

    actual = loss_function.forward_propagation(
        output,
        t,
    )

    expected = (
        F.binary_cross_entropy_with_logits(
            torch.tensor(
                output,
                dtype=torch.float64,
            ),
            torch.tensor(
                t,
                dtype=torch.float64,
            ),
            reduction="sum",
        )
        / output.shape[0]
    )

    assert np.isfinite(actual)

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    

def test_bce_with_logits_sigmoid_does_not_overflow():
    loss_function = BCEWithLogitsLoss()

    output = np.array([
        [1000.0],
        [-1000.0],
    ])

    t = np.array([
        [1.0],
        [0.0],
    ])

    with np.errstate(
        over="raise",
        invalid="raise",
    ):
        loss_function.forward_propagation(
            output,
            t,
        )

    assert np.all(
        np.isfinite(loss_function.sig_out)
    )
    
    
def test_bce_with_logits_backward_extreme_logits():
    loss_function = BCEWithLogitsLoss()

    output = np.array([
        [1000.0],
        [-1000.0],
    ])

    t = np.array([
        [0.0],
        [1.0],
    ])

    with np.errstate(over="ignore"):
        loss_function.forward_propagation(
            output,
            t,
        )

    actual = loss_function.backward_propagation()

    # sigmoid(1000) ≈ 1
    # sigmoid(-1000) ≈ 0
    expected = np.array([
        [0.5],
        [-0.5],
    ])

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_bce_with_logits_rejects_different_shapes():
    loss_function = BCEWithLogitsLoss()

    output = np.zeros((4, 3))
    t = np.zeros((4, 2))

    with pytest.raises(ValueError):
        loss_function.forward_propagation(
            output,
            t,
        )
        
        

