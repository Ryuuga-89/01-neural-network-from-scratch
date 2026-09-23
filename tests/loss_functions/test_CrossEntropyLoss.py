import numpy as np
import torch
import torch.nn.functional as F

from src.neural_networks.loss_functions.loss_functions import CrossEntropyLoss


def test_cross_entropy_forward():
    loss_function = CrossEntropyLoss()

    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.5, 0.3],
    ])

    target = np.array([0, 1])

    actual = loss_function.forward_propagation(
        logits,
        target,
    )

    # 数値安定化したsoftmax
    shifted = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )

    exp_logits = np.exp(shifted)

    probabilities = exp_logits / np.sum(
        exp_logits,
        axis=1,
        keepdims=True,
    )

    expected = -np.mean(
        np.log(
            probabilities[
                np.arange(len(target)),
                target,
            ]
        )
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    

def test_cross_entropy_forward_against_pytorch():
    loss_function = CrossEntropyLoss()

    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.5, 0.3],
        [-1.0, 0.2, 3.1],
    ])

    target = np.array([0, 1, 2])

    actual = loss_function.forward_propagation(
        logits,
        target,
    )

    logits_torch = torch.tensor(
        logits,
        dtype=torch.float64,
    )

    target_torch = torch.tensor(
        target,
        dtype=torch.long,
    )

    expected = F.cross_entropy(
        logits_torch,
        target_torch,
        reduction="mean",
    )

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_cross_entropy_backward():
    loss_function = CrossEntropyLoss()

    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.5, 0.3],
    ])

    target = np.array([0, 1])

    loss_function.forward_propagation(
        logits,
        target,
    )

    actual = loss_function.backward_propagation()

    shifted = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )

    exp_logits = np.exp(shifted)

    probabilities = exp_logits / np.sum(
        exp_logits,
        axis=1,
        keepdims=True,
    )

    expected = probabilities.copy()

    expected[
        np.arange(len(target)),
        target,
    ] -= 1.0

    expected /= len(target)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_cross_entropy_backward_against_pytorch():
    loss_function = CrossEntropyLoss()

    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.5, 0.3],
        [-1.0, 0.2, 3.1],
    ])

    target = np.array([0, 1, 2])

    # 自作
    loss_function.forward_propagation(
        logits,
        target,
    )

    actual = loss_function.backward_propagation()

    # PyTorch
    logits_torch = torch.tensor(
        logits,
        dtype=torch.float64,
        requires_grad=True,
    )

    target_torch = torch.tensor(
        target,
        dtype=torch.long,
    )

    loss_torch = F.cross_entropy(
        logits_torch,
        target_torch,
        reduction="mean",
    )

    loss_torch.backward()

    assert logits_torch.grad is not None
    expected = logits_torch.grad.numpy()

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    

def test_cross_entropy_backward_numerical_gradient():
    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.5, 0.3],
    ])

    target = np.array([0, 1])

    loss_function = CrossEntropyLoss()

    loss_function.forward_propagation(
        logits,
        target,
    )

    analytical = loss_function.backward_propagation()

    eps = 1e-5
    numerical = np.zeros_like(logits)

    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):

            original = logits[i, j]

            logits[i, j] = original + eps

            loss_plus = CrossEntropyLoss()
            l_plus = loss_plus.forward_propagation(
                logits,
                target,
            )

            logits[i, j] = original - eps

            loss_minus = CrossEntropyLoss()
            l_minus = loss_minus.forward_propagation(
                logits,
                target,
            )

            numerical[i, j] = (
                l_plus - l_minus
            ) / (2 * eps)

            logits[i, j] = original

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=1e-5,
        atol=1e-6,
    )
    
    
def test_cross_entropy_numerical_stability():
    loss_function = CrossEntropyLoss()

    logits = np.array([
        [1000.0, 1001.0, 999.0],
        [-1000.0, -999.0, -1001.0],
    ])

    target = np.array([1, 1])

    actual = loss_function.forward_propagation(
        logits,
        target,
    )

    logits_torch = torch.tensor(
        logits,
        dtype=torch.float64,
    )

    target_torch = torch.tensor(
        target,
        dtype=torch.long,
    )

    expected = F.cross_entropy(
        logits_torch,
        target_torch,
    )

    assert np.isfinite(actual)

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    