import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.neural_networks.layers.convolutional_layers import Conv2D

SEED = 42


def create_args(
    stride: int = 1,
    padding_mode: str = "Zeros",
    padding_length: int = 0,
) -> dict:
    return {
        "weight_init_method": {
            "method_name": "He",
            "distribution": "normal",
            "seed": SEED,
        },
        "bias_init_method": {
            "method_name": "Zeros",
        },
        "Conv2D": {
            "stride": stride,
            "padding_mode": padding_mode,
            "padding_length": padding_length,
        },
    }
    
    
def test_conv2d_initialization():
    conv = Conv2D(
        shape=(4, 3, 3, 3),
        args=create_args(),
    )

    assert conv.C_out == 4
    assert conv.C_in == 3
    assert conv.Kh == 3
    assert conv.Kw == 3

    assert conv.n_in == 27
    assert conv.n_out == 36

    assert conv.W.shape == (4, 3, 3, 3)
    assert conv.b.shape == (4,)
    

def test_conv2d_forward_known_values():
    conv = Conv2D(
        shape=(1, 1, 2, 2),
        args=create_args(),
    )

    conv.W[:] = np.array([
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ]
    ])

    conv.b[:] = np.array([1.0])

    x = np.array([
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ]
    ])

    actual = conv.forward_propagation(x)

    expected = np.array([
        [
            [
                [
                    1*1 + 2*2 + 4*3 + 5*4 + 1,
                    2*1 + 3*2 + 5*3 + 6*4 + 1,
                ],
                [
                    4*1 + 5*2 + 7*3 + 8*4 + 1,
                    5*1 + 6*2 + 8*3 + 9*4 + 1,
                ],
            ]
        ]
    ])

    assert actual.shape == (1, 1, 2, 2)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_conv2d_im2col():
    conv = Conv2D(
        shape=(1, 1, 2, 2),
        args=create_args(),
    )

    x = np.array([
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ]
    ])

    conv.x_shape = x.shape

    actual = conv.im2col(x)

    expected = np.array([
        [1.0, 2.0, 4.0, 5.0],
        [2.0, 3.0, 5.0, 6.0],
        [4.0, 5.0, 7.0, 8.0],
        [5.0, 6.0, 8.0, 9.0],
    ])

    assert actual.shape == (4, 4)

    np.testing.assert_array_equal(
        actual,
        expected,
    )
    
    
def test_conv2d_forward_against_pytorch():
    rng = np.random.default_rng(123)

    conv = Conv2D(
        shape=(3, 2, 3, 3),
        args=create_args(
            stride=1,
            padding_length=0,
        ),
    )

    x = rng.normal(
        size=(4, 2, 6, 7)
    )

    actual = conv.forward_propagation(x)

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
    )

    W_torch = torch.tensor(
        conv.W,
        dtype=torch.float64,
    )

    b_torch = torch.tensor(
        conv.b,
        dtype=torch.float64,
    )

    expected = F.conv2d(
        x_torch,
        W_torch,
        b_torch,
        stride=1,
        padding=0,
    )

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_conv2d_forward_stride_against_pytorch():
    rng = np.random.default_rng(123)

    conv = Conv2D(
        shape=(3, 2, 3, 3),
        args=create_args(
            stride=2,
            padding_length=0,
        ),
    )

    x = rng.normal(
        size=(2, 2, 7, 8)
    )

    actual = conv.forward_propagation(x)

    expected = F.conv2d(
        torch.tensor(x, dtype=torch.float64),
        torch.tensor(conv.W, dtype=torch.float64),
        torch.tensor(conv.b, dtype=torch.float64),
        stride=2,
        padding=0,
    )

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_conv2d_forward_zero_padding_against_pytorch():
    rng = np.random.default_rng(123)

    conv = Conv2D(
        shape=(3, 2, 3, 3),
        args=create_args(
            stride=1,
            padding_mode="Zeros",
            padding_length=1,
        ),
    )

    x = rng.normal(
        size=(2, 2, 5, 6)
    )

    actual = conv.forward_propagation(x)

    expected = F.conv2d(
        torch.tensor(x, dtype=torch.float64),
        torch.tensor(conv.W, dtype=torch.float64),
        torch.tensor(conv.b, dtype=torch.float64),
        stride=1,
        padding=1,
    )

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
def test_conv2d_forward_stride_and_padding_against_pytorch():
    rng = np.random.default_rng(123)

    conv = Conv2D(
        shape=(4, 2, 3, 3),
        args=create_args(
            stride=2,
            padding_mode="Zeros",
            padding_length=1,
        ),
    )

    x = rng.normal(
        size=(3, 2, 7, 8)
    )

    actual = conv.forward_propagation(x)

    expected = F.conv2d(
        torch.tensor(x, dtype=torch.float64),
        torch.tensor(conv.W, dtype=torch.float64),
        torch.tensor(conv.b, dtype=torch.float64),
        stride=2,
        padding=1,
    )

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_conv2d_backward_against_pytorch():
    rng = np.random.default_rng(123)

    conv = Conv2D(
        shape=(3, 2, 3, 3),
        args=create_args(
            stride=1,
            padding_length=0,
        ),
    )

    x = rng.normal(
        size=(2, 2, 5, 6)
    )

    # ---------- 自作 ----------

    out = conv.forward_propagation(x)

    dout = rng.normal(
        size=out.shape
    )

    actual_dx = conv.backward_propagation(dout)

    # ---------- PyTorch ----------

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
        requires_grad=True,
    )

    W_torch = torch.tensor(
        conv.W,
        dtype=torch.float64,
        requires_grad=True,
    )

    b_torch = torch.tensor(
        conv.b,
        dtype=torch.float64,
        requires_grad=True,
    )

    out_torch = F.conv2d(
        x_torch,
        W_torch,
        b_torch,
        stride=1,
        padding=0,
    )

    dout_torch = torch.tensor(
        dout,
        dtype=torch.float64,
    )

    out_torch.backward(dout_torch)

    assert x_torch.grad is not None
    assert W_torch.grad is not None
    assert b_torch.grad is not None

    np.testing.assert_allclose(
        actual_dx,
        x_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        conv.dW,
        W_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        conv.db,
        b_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_conv2d_backward_stride_padding_against_pytorch():
    rng = np.random.default_rng(123)

    conv = Conv2D(
        shape=(3, 2, 3, 3),
        args=create_args(
            stride=2,
            padding_mode="Zeros",
            padding_length=1,
        ),
    )

    x = rng.normal(
        size=(2, 2, 7, 8)
    )

    out = conv.forward_propagation(x)

    dout = rng.normal(
        size=out.shape
    )

    actual_dx = conv.backward_propagation(dout)

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
        requires_grad=True,
    )

    W_torch = torch.tensor(
        conv.W,
        dtype=torch.float64,
        requires_grad=True,
    )

    b_torch = torch.tensor(
        conv.b,
        dtype=torch.float64,
        requires_grad=True,
    )

    out_torch = F.conv2d(
        x_torch,
        W_torch,
        b_torch,
        stride=2,
        padding=1,
    )

    out_torch.backward(
        torch.tensor(
            dout,
            dtype=torch.float64,
        )
    )

    assert x_torch.grad is not None
    assert W_torch.grad is not None
    assert b_torch.grad is not None

    np.testing.assert_allclose(
        actual_dx,
        x_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        conv.dW,
        W_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        conv.db,
        b_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    

def test_conv2d_weight_numerical_gradient():
    rng = np.random.default_rng(123)

    conv = Conv2D(
        shape=(1, 1, 2, 2),
        args=create_args(),
    )

    x = rng.normal(
        size=(1, 1, 3, 3)
    )

    out = conv.forward_propagation(x)

    dout = rng.normal(
        size=out.shape
    )

    conv.backward_propagation(dout)

    analytical = conv.dW.copy()

    eps = 1e-5
    numerical = np.zeros_like(conv.W)

    for index in np.ndindex(conv.W.shape):
        original = conv.W[index]

        conv.W[index] = original + eps
        out_plus = conv.forward_propagation(x)
        loss_plus = np.sum(
            out_plus * dout
        )

        conv.W[index] = original - eps
        out_minus = conv.forward_propagation(x)
        loss_minus = np.sum(
            out_minus * dout
        )

        numerical[index] = (
            loss_plus - loss_minus
        ) / (2 * eps)

        conv.W[index] = original

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=1e-5,
        atol=1e-6,
    )
    
    
def test_conv2d_input_numerical_gradient():
    rng = np.random.default_rng(123)

    conv = Conv2D(
        shape=(1, 1, 2, 2),
        args=create_args(),
    )

    x = rng.normal(
        size=(1, 1, 3, 3)
    )

    out = conv.forward_propagation(x)

    dout = rng.normal(
        size=out.shape
    )

    analytical = conv.backward_propagation(
        dout
    )

    eps = 1e-5
    numerical = np.zeros_like(x)

    for index in np.ndindex(x.shape):
        original = x[index]

        x[index] = original + eps
        out_plus = conv.forward_propagation(x)
        loss_plus = np.sum(
            out_plus * dout
        )

        x[index] = original - eps
        out_minus = conv.forward_propagation(x)
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


@pytest.mark.parametrize(
    "padding_mode,np_mode",
    [
        ("Zeros", "constant"),
        ("Edge", "edge"),
        ("Reflect", "reflect"),
        ("Symmetric", "symmetric"),
    ],
)


def test_conv2d_padding_modes(
    padding_mode,
    np_mode,
):
    rng = np.random.default_rng(123)

    conv = Conv2D(
        shape=(2, 1, 3, 3),
        args=create_args(
            padding_mode=padding_mode,
            padding_length=1,
        ),
    )

    x = rng.normal(
        size=(2, 1, 5, 6)
    )

    actual = conv.forward_propagation(x)

    x_padded = np.pad(
        x,
        (
            (0, 0),
            (0, 0),
            (1, 1),
            (1, 1),
        ),
        mode=np_mode,
    )

    expected = F.conv2d(
        torch.tensor(
            x_padded,
            dtype=torch.float64,
        ),
        torch.tensor(
            conv.W,
            dtype=torch.float64,
        ),
        torch.tensor(
            conv.b,
            dtype=torch.float64,
        ),
        stride=1,
        padding=0,
    )

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )


def test_conv2d_edge_padding_backward_accumulates_to_original():
    conv = Conv2D(
        shape=(1, 1, 1, 1),
        args=create_args(
            stride=1,
            padding_mode="Edge",
            padding_length=1,
        ),
    )

    conv.W[:] = 1.0
    conv.b[:] = 0.0

    x = np.array([
        [
            [
                [5.0],
            ]
        ]
    ])

    out = conv.forward_propagation(x)

    # Edge paddingによって
    #
    # 5 5 5
    # 5 5 5
    # 5 5 5
    #
    # となる
    assert out.shape == (1, 1, 3, 3)

    dout = np.ones_like(out)

    actual = conv.backward_propagation(dout)

    # 9個すべての出力が元のx[0,0,0,0]に依存する
    expected = np.array([
        [
            [
                [9.0],
            ]
        ]
    ])

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    
    
@pytest.mark.parametrize(
    "padding_mode",
    [
        "Edge",
        "Reflect",
        "Symmetric",
    ],
)
@pytest.mark.parametrize(
    "padding_length",
    [1, 2],
)
def test_conv2d_padding_backward_numerical_gradient(
    padding_mode,
    padding_length,
):
    rng = np.random.default_rng(123)

    conv = Conv2D(
        shape=(1, 1, 2, 3),
        args=create_args(
            stride=1,
            padding_mode=padding_mode,
            padding_length=padding_length,
        ),
    )

    # Reflectでpadding_length=2を使えるサイズにする
    x = rng.normal(
        size=(1, 1, 4, 5)
    )

    out = conv.forward_propagation(x)

    dout = rng.normal(
        size=out.shape
    )

    analytical = conv.backward_propagation(
        dout
    )

    eps = 1e-5
    numerical = np.zeros_like(x)

    for index in np.ndindex(x.shape):
        original = x[index]

        x[index] = original + eps
        out_plus = conv.forward_propagation(x)
        loss_plus = np.sum(
            out_plus * dout
        )

        x[index] = original - eps
        out_minus = conv.forward_propagation(x)
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