from typing import Literal

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.neural_networks.layers.convolutional_layers import MaxPooling2D


def create_args(
    stride: int = 2,
    padding_mode: str = "Zeros",
    padding_length: int = 0,
) -> dict:
    return {
        "MaxPooling2D": {
            "stride": stride,
            "padding_mode": padding_mode,
            "padding_length": padding_length,
        }
    }
    

def test_maxpool_initialization():
    pool = MaxPooling2D(
        shape=(2, 3),
        args=create_args(stride=2),
    )

    assert pool.Kh == 2
    assert pool.Kw == 3
    assert pool.stride == 2
    assert pool.padding_mode == "Zeros"
    assert pool.padding_length == 0
    
    
def test_maxpool_forward_known_values():
    pool = MaxPooling2D(
        shape=(2, 2),
        args=create_args(stride=2),
    )

    x = np.array([
        [
            [
                [1.0, 2.0, 5.0, 4.0],
                [3.0, 4.0, 7.0, 6.0],
                [9.0, 8.0, 1.0, 2.0],
                [7.0, 6.0, 3.0, 4.0],
            ]
        ]
    ])

    actual = pool.forward_propagation(x)

    expected = np.array([
        [
            [
                [4.0, 7.0],
                [9.0, 4.0],
            ]
        ]
    ])

    assert actual.shape == (1, 1, 2, 2)

    np.testing.assert_array_equal(
        actual,
        expected,
    )
    

def test_maxpool_forward_multi_channel():
    pool = MaxPooling2D(
        shape=(2, 2),
        args=create_args(stride=2),
    )

    x = np.array([
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            [
                [8.0, 7.0],
                [6.0, 5.0],
            ],
        ]
    ])

    actual = pool.forward_propagation(x)

    expected = np.array([
        [
            [[4.0]],
            [[8.0]],
        ]
    ])

    assert actual.shape == (1, 2, 1, 1)

    np.testing.assert_array_equal(
        actual,
        expected,
    )
    
    
def test_maxpool_forward_against_pytorch():
    rng = np.random.default_rng(123)

    pool = MaxPooling2D(
        shape=(2, 2),
        args=create_args(
            stride=2,
            padding_length=0,
        ),
    )

    x = rng.normal(
        size=(3, 4, 6, 8)
    )

    actual = pool.forward_propagation(x)

    expected = F.max_pool2d(
        torch.tensor(x, dtype=torch.float64),
        kernel_size=(2, 2),
        stride=2,
        padding=0,
    )

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_maxpool_rectangular_kernel_against_pytorch():
    rng = np.random.default_rng(123)

    pool = MaxPooling2D(
        shape=(2, 3),
        args=create_args(stride=2),
    )

    x = rng.normal(
        size=(2, 3, 7, 8)
    )

    actual = pool.forward_propagation(x)

    expected = F.max_pool2d(
        torch.tensor(x, dtype=torch.float64),
        kernel_size=(2, 3),
        stride=2,
    )

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    

def test_maxpool_backward_known_values():
    pool = MaxPooling2D(
        shape=(2, 2),
        args=create_args(stride=2),
    )

    x = np.array([
        [
            [
                [1.0, 2.0, 5.0, 4.0],
                [3.0, 4.0, 7.0, 6.0],
                [9.0, 8.0, 1.0, 2.0],
                [7.0, 6.0, 3.0, 4.0],
            ]
        ]
    ])

    pool.forward_propagation(x)

    dout = np.array([
        [
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ]
        ]
    ])

    actual = pool.backward_propagation(dout)

    expected = np.array([
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 10.0, 20.0, 0.0],
                [30.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 40.0],
            ]
        ]
    ])

    np.testing.assert_array_equal(
        actual,
        expected,
    )
    
    
def test_maxpool_backward_against_pytorch():
    rng = np.random.default_rng(123)

    pool = MaxPooling2D(
        shape=(2, 2),
        args=create_args(stride=2),
    )

    # 同値最大値が偶然発生しにくい連続乱数
    x = rng.normal(
        size=(2, 3, 6, 8)
    )

    out = pool.forward_propagation(x)

    dout = rng.normal(
        size=out.shape
    )

    actual = pool.backward_propagation(dout)

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
        requires_grad=True,
    )

    out_torch = F.max_pool2d(
        x_torch,
        kernel_size=(2, 2),
        stride=2,
    )

    out_torch.backward(
        torch.tensor(
            dout,
            dtype=torch.float64,
        )
    )

    assert x_torch.grad is not None

    np.testing.assert_allclose(
        actual,
        x_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_maxpool_backward_accumulates_overlapping_gradients():
    pool = MaxPooling2D(
        shape=(2, 2),
        args=create_args(stride=1),
    )

    x = np.array([
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 9.0, 5.0],
                [6.0, 7.0, 8.0],
            ]
        ]
    ])

    out = pool.forward_propagation(x)

    assert out.shape == (1, 1, 2, 2)

    dout = np.ones_like(out)

    actual = pool.backward_propagation(dout)

    expected = np.array([
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ]
    ])

    np.testing.assert_array_equal(
        actual,
        expected,
    )
    
    
def test_maxpool_stride_one_against_pytorch():
    rng = np.random.default_rng(123)

    pool = MaxPooling2D(
        shape=(3, 3),
        args=create_args(stride=1),
    )

    x = rng.normal(
        size=(2, 3, 6, 7)
    )

    actual = pool.forward_propagation(x)

    expected = F.max_pool2d(
        torch.tensor(x, dtype=torch.float64),
        kernel_size=3,
        stride=1,
    )

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
def test_maxpool_zero_padding_uses_negative_infinity():
    pool = MaxPooling2D(
        shape=(3, 3),
        args=create_args(
            stride=1,
            padding_mode="Zeros",
            padding_length=1,
        ),
    )

    x = np.array([
        [
            [
                [-5.0, -4.0],
                [-3.0, -2.0],
            ]
        ]
    ])

    actual = pool.forward_propagation(x)

    expected = F.max_pool2d(
        torch.tensor(
            x,
            dtype=torch.float64,
        ),
        kernel_size=3,
        stride=1,
        padding=1,
    )

    # paddingの0が最大値として選ばれてはいけない
    assert np.all(actual < 0)

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    

def test_maxpool_backward_with_padding_against_pytorch():
    rng = np.random.default_rng(123)

    pool = MaxPooling2D(
        shape=(3, 3),
        args=create_args(
            stride=1,
            padding_mode="Zeros",
            padding_length=1,
        ),
    )

    x = rng.normal(
        size=(2, 2, 5, 6)
    )

    out = pool.forward_propagation(x)

    dout = rng.normal(
        size=out.shape
    )

    actual = pool.backward_propagation(dout)

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
        requires_grad=True,
    )

    out_torch = F.max_pool2d(
        x_torch,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    out_torch.backward(
        torch.tensor(
            dout,
            dtype=torch.float64,
        )
    )

    assert x_torch.grad is not None

    np.testing.assert_allclose(
        actual,
        x_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
@pytest.mark.parametrize(
    "padding_mode,np_mode",
    [
        ("Edge", "edge"),
        ("Reflect", "reflect"),
        ("Symmetric", "symmetric"),
    ],
)
def test_maxpool_padding_modes_forward(
    padding_mode,
    np_mode,
):
    rng = np.random.default_rng(123)

    pool = MaxPooling2D(
        shape=(3, 3),
        args=create_args(
            stride=1,
            padding_mode=padding_mode,
            padding_length=1,
        ),
    )

    x = rng.normal(
        size=(2, 2, 5, 6)
    )

    actual = pool.forward_propagation(x)

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

    expected = F.max_pool2d(
        torch.tensor(
            x_padded,
            dtype=torch.float64,
        ),
        kernel_size=3,
        stride=1,
        padding=0,
    )

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_maxpool_edge_padding_backward_accumulates_to_original():
    pool = MaxPooling2D(
        shape=(1, 1),
        args=create_args(
            stride=1,
            padding_mode="Edge",
            padding_length=1,
        ),
    )

    x = np.array([
        [
            [
                [5.0],
            ]
        ]
    ])

    out = pool.forward_propagation(x)

    assert out.shape == (1, 1, 3, 3)

    expected_out = np.full(
        (1, 1, 3, 3),
        5.0,
    )

    np.testing.assert_array_equal(
        out,
        expected_out,
    )

    dout = np.ones_like(out)

    actual = pool.backward_propagation(dout)

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
    
PaddingMode = Literal[
    "edge",
    "reflect",
    "symmetric",
]
    
def pad_by_index_torch(
    x: torch.Tensor,
    padding_length: int,
    np_mode: PaddingMode,
) -> torch.Tensor:
    H = x.shape[2]
    W = x.shape[3]

    row_indices = np.pad(
        np.arange(H),
        pad_width=(padding_length, padding_length),
        mode=np_mode,
    )

    col_indices = np.pad(
        np.arange(W),
        pad_width=(padding_length, padding_length),
        mode=np_mode,
    )

    row_indices_torch = torch.tensor(
        row_indices,
        dtype=torch.long,
    )

    col_indices_torch = torch.tensor(
        col_indices,
        dtype=torch.long,
    )

    return (
        x
        .index_select(2, row_indices_torch)
        .index_select(3, col_indices_torch)
    )
    
    
@pytest.mark.parametrize(
    "padding_mode,np_mode",
    [
        ("Edge", "edge"),
        ("Reflect", "reflect"),
        ("Symmetric", "symmetric"),
    ],
)
@pytest.mark.parametrize(
    "padding_length",
    [1, 2],
)
@pytest.mark.parametrize(
    "stride",
    [1, 2],
)
def test_maxpool_padding_backward_against_pytorch(
    padding_mode,
    np_mode,
    padding_length,
    stride,
):
    rng = np.random.default_rng(123)

    pool = MaxPooling2D(
        shape=(2, 3),
        args=create_args(
            stride=stride,
            padding_mode=padding_mode,
            padding_length=padding_length,
        ),
    )

    # Reflectでpadding=2も使用できる十分なサイズ
    x = rng.normal(
        size=(2, 2, 5, 6)
    )

    # =====================
    # 自作実装
    # =====================

    out = pool.forward_propagation(x)

    dout = rng.normal(
        size=out.shape
    )

    actual_dx = pool.backward_propagation(
        dout
    )

    # =====================
    # PyTorch参照実装
    # =====================

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
        requires_grad=True,
    )

    x_padded_torch = pad_by_index_torch(
        x_torch,
        padding_length,
        np_mode,
    )

    out_torch = F.max_pool2d(
        x_padded_torch,
        kernel_size=(2, 3),
        stride=stride,
        padding=0,
    )

    # forwardも同時に確認
    np.testing.assert_allclose(
        out,
        out_torch.detach().numpy(),
        rtol=1e-7,
        atol=1e-7,
    )

    out_torch.backward(
        torch.tensor(
            dout,
            dtype=torch.float64,
        )
    )

    assert x_torch.grad is not None

    # paddingの逆伝播を含むdxを比較
    np.testing.assert_allclose(
        actual_dx,
        x_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
def test_maxpool_backward_stride_and_zero_padding_against_pytorch():
    rng = np.random.default_rng(123)

    pool = MaxPooling2D(
        shape=(3, 3),
        args=create_args(
            stride=2,
            padding_mode="Zeros",
            padding_length=1,
        ),
    )

    x = rng.normal(
        size=(2, 2, 7, 8)
    )

    out = pool.forward_propagation(x)

    dout = rng.normal(
        size=out.shape
    )

    actual = pool.backward_propagation(
        dout
    )

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
        requires_grad=True,
    )

    out_torch = F.max_pool2d(
        x_torch,
        kernel_size=(3, 3),
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

    np.testing.assert_allclose(
        actual,
        x_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )