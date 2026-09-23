import numpy as np
import pytest
import torch

from src.neural_networks.layers.other_layers import BatchNormalization


def create_args(
    channels: int,
    momentum: float = 0.9,
    eps: float = 1e-5,
) -> dict:
    return {
        "BatchNormalization": {
            "channels": channels,
            "momentum": momentum,
            "eps": eps,
        }
    }
    
    
def test_batchnorm_initialization():
    bn = BatchNormalization(
        args=create_args(
            channels=3,
            momentum=0.9,
            eps=1e-5,
        )
    )

    assert bn.C == 3
    assert bn.momentum == 0.9
    assert bn.eps == 1e-5

    np.testing.assert_array_equal(
        bn.gamma,
        np.ones(3),
    )

    np.testing.assert_array_equal(
        bn.beta,
        np.zeros(3),
    )

    np.testing.assert_array_equal(
        bn.mu_infer,
        np.zeros(3),
    )

    np.testing.assert_array_equal(
        bn.var_infer,
        np.ones(3),
    )
    

def test_batchnorm_forward_2d():
    bn = BatchNormalization(
        args=create_args(
            channels=3,
            eps=1e-5,
        )
    )

    x = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0],
        [4.0, 8.0, 12.0],
    ])

    bn.gamma[:] = np.array([
        1.5, 0.5, 2.0,
    ])

    bn.beta[:] = np.array([
        0.1, -0.2, 0.3,
    ])

    actual = bn.forward_propagation(x)

    mu = np.mean(
        x,
        axis=0,
        keepdims=True,
    )

    var = np.mean(
        (x - mu) ** 2,
        axis=0,
        keepdims=True,
    )

    xn = (
        x - mu
    ) / np.sqrt(var + bn.eps)

    expected = (
        bn.gamma.reshape(1, -1) * xn
        + bn.beta.reshape(1, -1)
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_batchnorm_forward_4d():
    rng = np.random.default_rng(123)

    bn = BatchNormalization(
        args=create_args(
            channels=3,
            eps=1e-5,
        )
    )

    x = rng.normal(
        size=(4, 3, 5, 6)
    )

    bn.gamma[:] = np.array([
        1.2, 0.7, 2.0,
    ])

    bn.beta[:] = np.array([
        0.1, -0.3, 0.5,
    ])

    actual = bn.forward_propagation(x)

    mu = np.mean(
        x,
        axis=(0, 2, 3),
        keepdims=True,
    )

    var = np.mean(
        (x - mu) ** 2,
        axis=(0, 2, 3),
        keepdims=True,
    )

    xn = (
        x - mu
    ) / np.sqrt(var + bn.eps)

    expected = (
        bn.gamma.reshape(1, -1, 1, 1)
        * xn
        + bn.beta.reshape(1, -1, 1, 1)
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_batchnorm_forward_2d_against_pytorch():
    rng = np.random.default_rng(123)

    bn = BatchNormalization(
        args=create_args(
            channels=3,
            eps=1e-5,
        )
    )

    x = rng.normal(
        size=(8, 3)
    )

    bn.gamma[:] = np.array([
        1.2, 0.7, 2.0,
    ])

    bn.beta[:] = np.array([
        0.1, -0.3, 0.5,
    ])

    actual = bn.forward_propagation(x)

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
    )

    torch_bn = torch.nn.BatchNorm1d(
        3,
        eps=bn.eps,
        affine=True,
        track_running_stats=False,
    ).double()

    with torch.no_grad():
        torch_bn.weight.copy_(
            torch.tensor(
                bn.gamma,
                dtype=torch.float64,
            )
        )

        torch_bn.bias.copy_(
            torch.tensor(
                bn.beta,
                dtype=torch.float64,
            )
        )

    expected = torch_bn(x_torch)

    np.testing.assert_allclose(
        actual,
        expected.detach().numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
def test_batchnorm_forward_4d_against_pytorch():
    rng = np.random.default_rng(123)

    bn = BatchNormalization(
        args=create_args(
            channels=3,
            eps=1e-5,
        )
    )

    x = rng.normal(
        size=(4, 3, 5, 6)
    )

    actual = bn.forward_propagation(x)

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
    )

    torch_bn = torch.nn.BatchNorm2d(
        3,
        eps=bn.eps,
        affine=True,
        track_running_stats=False,
    ).double()

    with torch.no_grad():
        torch_bn.weight.copy_(
            torch.tensor(
                bn.gamma,
                dtype=torch.float64,
            )
        )

        torch_bn.bias.copy_(
            torch.tensor(
                bn.beta,
                dtype=torch.float64,
            )
        )

    expected = torch_bn(x_torch)

    np.testing.assert_allclose(
        actual,
        expected.detach().numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
def test_batchnorm_updates_running_statistics():
    momentum = 0.9

    bn = BatchNormalization(
        args=create_args(
            channels=2,
            momentum=momentum,
        )
    )

    x = np.array([
        [1.0, 2.0],
        [3.0, 6.0],
        [5.0, 10.0],
        [7.0, 14.0],
    ])

    old_mu = bn.mu_infer.copy()
    old_var = bn.var_infer.copy()

    batch_mu = np.mean(
        x,
        axis=0,
    )

    batch_var = np.mean(
        (x - batch_mu) ** 2,
        axis=0,
    )

    bn.forward_propagation(x)

    expected_mu = (
        momentum * old_mu
        + (1.0 - momentum) * batch_mu
    )

    expected_var = (
        momentum * old_var
        + (1.0 - momentum) * batch_var
    )

    np.testing.assert_allclose(
        bn.mu_infer,
        expected_mu,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        bn.var_infer,
        expected_var,
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_batchnorm_eval_uses_running_statistics():
    bn = BatchNormalization(
        args=create_args(
            channels=2,
            eps=1e-5,
        )
    )

    bn.mu_infer[:] = np.array([
        2.0,
        -1.0,
    ])

    bn.var_infer[:] = np.array([
        4.0,
        9.0,
    ])

    bn.gamma[:] = np.array([
        2.0,
        0.5,
    ])

    bn.beta[:] = np.array([
        1.0,
        -2.0,
    ])

    bn.train = False

    x = np.array([
        [4.0, 2.0],
        [6.0, 5.0],
    ])

    actual = bn.forward_propagation(x)

    xn = (
        x - bn.mu_infer.reshape(1, -1)
    ) / np.sqrt(
        bn.var_infer.reshape(1, -1)
        + bn.eps
    )

    expected = (
        bn.gamma.reshape(1, -1) * xn
        + bn.beta.reshape(1, -1)
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_batchnorm_eval_does_not_update_running_statistics():
    bn = BatchNormalization(
        args=create_args(
            channels=2,
        )
    )

    bn.mu_infer[:] = np.array([
        1.0, 2.0,
    ])

    bn.var_infer[:] = np.array([
        3.0, 4.0,
    ])

    old_mu = bn.mu_infer.copy()
    old_var = bn.var_infer.copy()

    bn.train = False

    x = np.array([
        [100.0, -100.0],
        [200.0, -200.0],
    ])

    bn.forward_propagation(x)

    np.testing.assert_array_equal(
        bn.mu_infer,
        old_mu,
    )

    np.testing.assert_array_equal(
        bn.var_infer,
        old_var,
    )
    
def test_batchnorm_backward_2d_against_pytorch():
    rng = np.random.default_rng(123)

    bn = BatchNormalization(
        args=create_args(
            channels=3,
            eps=1e-5,
        )
    )

    x = rng.normal(
        size=(6, 3)
    )

    bn.gamma[:] = np.array([
        1.2, 0.7, 2.0,
    ])

    bn.beta[:] = np.array([
        0.1, -0.3, 0.5,
    ])

    out = bn.forward_propagation(x)

    dout = rng.normal(
        size=out.shape
    )

    actual_dx = bn.backward_propagation(
        dout
    )

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
        requires_grad=True,
    )

    gamma_torch = torch.tensor(
        bn.gamma,
        dtype=torch.float64,
        requires_grad=True,
    )

    beta_torch = torch.tensor(
        bn.beta,
        dtype=torch.float64,
        requires_grad=True,
    )

    mu = x_torch.mean(
        dim=0,
        keepdim=True,
    )

    var = (
        (x_torch - mu) ** 2
    ).mean(
        dim=0,
        keepdim=True,
    )

    xn = (
        x_torch - mu
    ) / torch.sqrt(
        var + bn.eps
    )

    out_torch = (
        gamma_torch.reshape(1, -1) * xn
        + beta_torch.reshape(1, -1)
    )

    out_torch.backward(
        torch.tensor(
            dout,
            dtype=torch.float64,
        )
    )

    assert x_torch.grad is not None
    assert gamma_torch.grad is not None
    assert beta_torch.grad is not None

    np.testing.assert_allclose(
        actual_dx,
        x_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        bn.dgamma,
        gamma_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        bn.dbeta,
        beta_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_batchnorm_backward_4d_against_pytorch():
    rng = np.random.default_rng(123)

    bn = BatchNormalization(
        args=create_args(
            channels=3,
            eps=1e-5,
        )
    )

    x = rng.normal(
        size=(4, 3, 5, 6)
    )

    bn.gamma[:] = np.array([
        1.2, 0.7, 2.0,
    ])

    out = bn.forward_propagation(x)

    dout = rng.normal(
        size=out.shape
    )

    actual_dx = bn.backward_propagation(
        dout
    )

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
        requires_grad=True,
    )

    gamma_torch = torch.tensor(
        bn.gamma,
        dtype=torch.float64,
        requires_grad=True,
    )

    beta_torch = torch.tensor(
        bn.beta,
        dtype=torch.float64,
        requires_grad=True,
    )

    mu = x_torch.mean(
        dim=(0, 2, 3),
        keepdim=True,
    )

    var = (
        (x_torch - mu) ** 2
    ).mean(
        dim=(0, 2, 3),
        keepdim=True,
    )

    xn = (
        x_torch - mu
    ) / torch.sqrt(
        var + bn.eps
    )

    out_torch = (
        gamma_torch.reshape(
            1, -1, 1, 1
        ) * xn
        + beta_torch.reshape(
            1, -1, 1, 1
        )
    )

    out_torch.backward(
        torch.tensor(
            dout,
            dtype=torch.float64,
        )
    )

    assert x_torch.grad is not None
    assert gamma_torch.grad is not None
    assert beta_torch.grad is not None

    np.testing.assert_allclose(
        actual_dx,
        x_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        bn.dgamma,
        gamma_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        bn.dbeta,
        beta_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_batchnorm_input_numerical_gradient():
    rng = np.random.default_rng(123)

    bn = BatchNormalization(
        args=create_args(
            channels=2,
            eps=1e-5,
        )
    )

    x = rng.normal(
        size=(4, 2)
    )

    out = bn.forward_propagation(x)

    dout = rng.normal(
        size=out.shape
    )

    analytical = bn.backward_propagation(
        dout
    )

    eps = 1e-5
    numerical = np.zeros_like(x)

    for index in np.ndindex(x.shape):
        original = x[index]

        x[index] = original + eps

        bn_plus = BatchNormalization(
            args=create_args(
                channels=2,
                eps=bn.eps,
            )
        )

        bn_plus.gamma[:] = bn.gamma
        bn_plus.beta[:] = bn.beta

        out_plus = bn_plus.forward_propagation(
            x
        )

        loss_plus = np.sum(
            out_plus * dout
        )

        x[index] = original - eps

        bn_minus = BatchNormalization(
            args=create_args(
                channels=2,
                eps=bn.eps,
            )
        )

        bn_minus.gamma[:] = bn.gamma
        bn_minus.beta[:] = bn.beta

        out_minus = bn_minus.forward_propagation(
            x
        )

        loss_minus = np.sum(
            out_minus * dout
        )

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
    
    
def test_batchnorm_gamma_beta_numerical_gradient():
    rng = np.random.default_rng(123)

    bn = BatchNormalization(
        args=create_args(
            channels=2,
            eps=1e-5,
        )
    )

    x = rng.normal(
        size=(4, 2)
    )

    out = bn.forward_propagation(x)

    dout = rng.normal(
        size=out.shape
    )

    bn.backward_propagation(dout)

    analytical_dgamma = bn.dgamma.copy()
    analytical_dbeta = bn.dbeta.copy()

    eps = 1e-5

    numerical_dgamma = np.zeros_like(
        bn.gamma
    )

    numerical_dbeta = np.zeros_like(
        bn.beta
    )

    for i in range(bn.C):
        original = bn.gamma[i]

        bn.gamma[i] = original + eps
        out_plus = bn.forward_propagation(x)
        loss_plus = np.sum(
            out_plus * dout
        )

        bn.gamma[i] = original - eps
        out_minus = bn.forward_propagation(x)
        loss_minus = np.sum(
            out_minus * dout
        )

        numerical_dgamma[i] = (
            loss_plus - loss_minus
        ) / (2 * eps)

        bn.gamma[i] = original

    for i in range(bn.C):
        original = bn.beta[i]

        bn.beta[i] = original + eps
        out_plus = bn.forward_propagation(x)
        loss_plus = np.sum(
            out_plus * dout
        )

        bn.beta[i] = original - eps
        out_minus = bn.forward_propagation(x)
        loss_minus = np.sum(
            out_minus * dout
        )

        numerical_dbeta[i] = (
            loss_plus - loss_minus
        ) / (2 * eps)

        bn.beta[i] = original

    np.testing.assert_allclose(
        analytical_dgamma,
        numerical_dgamma,
        rtol=1e-5,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        analytical_dbeta,
        numerical_dbeta,
        rtol=1e-5,
        atol=1e-6,
    )
    
    
def test_batchnorm_rejects_invalid_dimension():
    bn = BatchNormalization(
        args=create_args(
            channels=3,
        )
    )

    x = np.zeros(
        (2, 3, 4)
    )

    with pytest.raises(
        ValueError,
        match="BatchNorm expects 2D or 4D tensor",
    ):
        bn.forward_propagation(x)
        
        
        
    