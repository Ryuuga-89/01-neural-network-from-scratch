import numpy as np
import torch

from src.neural_networks.layers.basic_layers import Linear

# Heの初期化時のテスト
TEST_LINEAR_SHAPE1: tuple = (2, 3)
TEST_LINEAR_ARGS1: dict = {
    "weight_init_method": {
        "method_name": "He",
        "distribution": "normal",
        "seed": 42
    },
    
    "bias_init_method": {
        "method_name": "Zeros"
    }
}

# Xavierの初期化時のテスト
TEST_LINEAR_SHAPE2: tuple = (1, 2)
TEST_LINEAR_ARGS2: dict = {
    "weight_init_method": {
        "method_name": "Xavier",
        "distribution": "uniform",
        "seed": 42
    },
    
    "bias_init_method": {
        "method_name": "Zeros"
    }
}

# Standardの初期化時のテスト
TEST_LINEAR_SHAPE3: tuple = (100, 200)
TEST_LINEAR_ARGS3: dict = {
    "weight_init_method": {
        "method_name": "Standard",
        "distribution": "uniform",
        "r": 1.0,
        "seed": 42
    },
    
    "bias_init_method": {
        "method_name": "Zeros"
    }
}

# Standardの初期化時のテスト
TEST_LINEAR_SHAPE4: tuple = (10, 10)
TEST_LINEAR_ARGS4: dict = {
    "weight_init_method": {
        "method_name": "Standard",
        "distribution": "normal",
        "sigma": 1.0,
        "seed": 42
    },
    
    "bias_init_method": {
        "method_name": "Zeros"
    }
}

TEST_LINEAR_SHAPE_LIST: list[tuple] = [TEST_LINEAR_SHAPE1, TEST_LINEAR_SHAPE2, TEST_LINEAR_SHAPE3, TEST_LINEAR_SHAPE4]
TEST_LINEAR_ARGS_LIST: list[dict] = [TEST_LINEAR_ARGS1, TEST_LINEAR_ARGS2, TEST_LINEAR_ARGS3, TEST_LINEAR_ARGS4]


def test_linear_initialization():
    """
    Linearの初期化についてテストする
    """
    
    for i in range(4):
        shape = TEST_LINEAR_SHAPE_LIST[i]
        args = TEST_LINEAR_ARGS_LIST[i]
        test_linear: Linear = Linear(shape=shape, args=args)
        
        assert test_linear.W.shape == shape
        assert test_linear.b.shape == (shape[1],)
        assert test_linear.n_in == shape[0]
        assert test_linear.n_out == shape[1]
        
        assert isinstance(test_linear, Linear)
    

"""
Linear(shape=TEST_LINEAR_SHAPE1, args=TEST_LINEAR_ARGS1)にてインスタンス化すると

W =
[[ 0.30471708 -1.03998411  0.7504512 ]
 [ 0.94056472 -1.95103519 -1.30217951]]
 
b = [0. 0. 0.]

で初期化される

2, 3, 4次元の入力に対する出力を確認する。
入力に使う要素は全て1であるとする。
この時出力の最後の次元を取り出すと全て[1.2452818, -2.9910193, -0.55172831]になる
"""
def test_linear_forward():
    linear = Linear(shape=TEST_LINEAR_SHAPE1, args=TEST_LINEAR_ARGS1)
    expected_vec = np.array([1.2452818, -2.9910193, -0.55172831])

    test_shape_list: list[tuple] = [(5, 2), (3, 2, 2), (10, 4, 382, 2)]
    for shape in test_shape_list:
        x = np.ones(shape=shape, dtype=float)
        out = linear.forward_propagation(x)

        # 期待値テンソルを入力形状に合わせて broadcast
        expected_shape = shape[:-1] + (3,)
        expected = np.broadcast_to(expected_vec, expected_shape)

        # 形状の検証
        assert out.shape == expected_shape
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
        

"""
同様に逆伝播もテスト
x = 全て1
dout = 全て1
の条件下でテストする
"""
def test_linear_backward():
    W = np.array(
        [
            [0.30471708, -1.03998411, 0.7504512],
            [0.94056472, -1.95103519, -1.30217951],
        ]
    )
    b = np.array([0.0, 0.0, 0.0])

    # dx の各成分がとる期待値ベクトル (1, 3) @ W.T
    expected_dx_vec = np.sum(W, axis=1)  # array([0.01518417, -2.31265])

    test_shape_list: list[tuple] = [(5, 2), (3, 2, 2), (10, 4, 382, 2)]

    for shape in test_shape_list:
        linear = Linear(shape=TEST_LINEAR_SHAPE1, args=TEST_LINEAR_ARGS1)

        # 順伝播でキャッシュを保持
        x = np.ones(shape=shape, dtype=float)
        _ = linear.forward_propagation(x)

        # 上流からの勾配
        dy_shape = shape[:-1] + (3,)
        dout = np.ones(shape=dy_shape, dtype=float)

        # 逆伝播を実行
        dx = linear.backward_propagation(dout)

        assert dx.shape == shape
        expected_dx = np.broadcast_to(expected_dx_vec, shape)
        np.testing.assert_allclose(dx, expected_dx, rtol=1e-5, atol=1e-6)

        n_samples = int(np.prod(shape[:-1]))

        expected_dw = np.full((2, 3), fill_value=float(n_samples))
        expected_db = np.full((3,), fill_value=float(n_samples))

        np.testing.assert_allclose(linear.dW, expected_dw, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(linear.db, expected_db, rtol=1e-5, atol=1e-6)
    
    
def test_linear_backward_numerical_gradient():
    """
    Linear.backward_propagation() で計算される dW が
    数値微分と一致することを確認する。
    """
    linear = Linear(
        shape=TEST_LINEAR_SHAPE1,
        args=TEST_LINEAR_ARGS1,
    )

    x = np.array([
        [0.2, -0.3],
        [0.4, 0.1],
    ])

    # ----- analytical gradient -----

    y = linear.forward_propagation(x)

    # L = sum(y^2)
    # dL/dy = 2y
    dout = 2.0 * y

    linear.backward_propagation(dout)

    analytical_dW = linear.dW.copy()

    # 数値微分

    eps = 1e-5
    numerical_dW = np.zeros_like(linear.W)

    for i in range(linear.W.shape[0]):
        for j in range(linear.W.shape[1]):

            original = linear.W[i, j]

            # W_ij + eps
            linear.W[i, j] = original + eps
            y_plus = linear.forward_propagation(x)
            loss_plus = np.sum(y_plus ** 2)

            # W_ij - eps
            linear.W[i, j] = original - eps
            y_minus = linear.forward_propagation(x)
            loss_minus = np.sum(y_minus ** 2)

            numerical_dW[i, j] = (
                loss_plus - loss_minus
            ) / (2 * eps)

            # 元に戻す
            linear.W[i, j] = original

    np.testing.assert_allclose(
        analytical_dW,
        numerical_dW,
        rtol=1e-5,
        atol=1e-6,
    )
    
    
def test_linear_backward_bias_numerical_gradient():
    linear = Linear(
        shape=TEST_LINEAR_SHAPE1,
        args=TEST_LINEAR_ARGS1,
    )

    x = np.array([
        [0.2, -0.3],
        [0.4, 0.1],
    ])

    y = linear.forward_propagation(x)
    dout = 2.0 * y

    linear.backward_propagation(dout)

    analytical_db = linear.db.copy()

    eps = 1e-5
    numerical_db = np.zeros_like(linear.b)

    for i in range(linear.b.shape[0]):
        original = linear.b[i]

        linear.b[i] = original + eps
        y_plus = linear.forward_propagation(x)
        loss_plus = np.sum(y_plus ** 2)

        linear.b[i] = original - eps
        y_minus = linear.forward_propagation(x)
        loss_minus = np.sum(y_minus ** 2)

        numerical_db[i] = (
            loss_plus - loss_minus
        ) / (2 * eps)

        linear.b[i] = original

    np.testing.assert_allclose(
        analytical_db,
        numerical_db,
        rtol=1e-5,
        atol=1e-6,
    )


def test_linear_forward_against_pytorch():
    linear = Linear(
        shape=TEST_LINEAR_SHAPE1,
        args=TEST_LINEAR_ARGS1,
    )

    x = np.array([
        [0.2, -0.3],
        [0.4, 0.1],
    ])

    actual = linear.forward_propagation(x)

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
    )

    W_torch = torch.tensor(
        linear.W,
        dtype=torch.float64,
    )

    b_torch = torch.tensor(
        linear.b,
        dtype=torch.float64,
    )

    expected = x_torch @ W_torch + b_torch

    np.testing.assert_allclose(
        actual,
        expected.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_linear_backward_against_pytorch():
    linear = Linear(
        shape=TEST_LINEAR_SHAPE1,
        args=TEST_LINEAR_ARGS1,
    )

    x = np.array([
        [0.2, -0.3],
        [0.4, 0.1],
    ])

    # 自作

    y = linear.forward_propagation(x)

    dout = np.array([
        [0.1, -0.2, 0.3],
        [-0.4, 0.5, 0.6],
    ])

    dx = linear.backward_propagation(dout)

    # PyTorch

    x_torch = torch.tensor(
        x,
        dtype=torch.float64,
        requires_grad=True,
    )

    W_torch = torch.tensor(
        linear.W,
        dtype=torch.float64,
        requires_grad=True,
    )

    b_torch = torch.tensor(
        linear.b,
        dtype=torch.float64,
        requires_grad=True,
    )

    y_torch = x_torch @ W_torch + b_torch

    dout_torch = torch.tensor(
        dout,
        dtype=torch.float64,
    )

    y_torch.backward(dout_torch)

    # 比較

    assert x_torch.grad is not None
    np.testing.assert_allclose(
        dx,
        x_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )

    assert W_torch.grad is not None
    np.testing.assert_allclose(
        linear.dW,
        W_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )

    assert b_torch.grad is not None
    np.testing.assert_allclose(
        linear.db,
        b_torch.grad.numpy(),
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_he_initialization_statistics():
    args = {
        "weight_init_method": {
            "method_name": "He",
            "distribution": "normal",
            "seed": 42,
        },
        "bias_init_method": {
            "method_name": "Zeros",
        },
    }

    linear = Linear(
        shape=(1000, 1000),
        args=args,
    )

    expected_std = np.sqrt(2 / 1000)

    assert np.isclose(
        np.std(linear.W),
        expected_std,
        rtol=0.05,
    )

    assert np.isclose(
        np.mean(linear.W),
        0.0,
        atol=0.005,
    )

    np.testing.assert_array_equal(
        linear.b,
        np.zeros(1000),
    )


# インスタンス化時の乱数を確認するため
if __name__ == "__main__":
    shape = TEST_LINEAR_SHAPE1
    args = TEST_LINEAR_ARGS1
    
    linear = Linear(shape=shape, args=args)
    
    print(linear.W)
    print(linear.b)