import numpy as np
import torch

from src.neural_networks.layers.basic_layers import Linear
from src.neural_networks.layers.other_layers import Sequential
from src.neural_networks.optimizers.optimizers import Momentum

LINEAR_ARGS = {
    "weight_init_method": {
        "method_name": "He",
        "distribution": "normal",
        "seed": 42,
    },
    "bias_init_method": {
        "method_name": "Zeros",
    },
}


def create_model() -> Sequential:
    return Sequential([
        Linear(
            shape=(2, 3),
            args=LINEAR_ARGS,
        ),
    ])


def get_linear(model: Sequential) -> Linear:
    linear = model.layers[0]
    assert isinstance(linear, Linear)
    return linear


def test_momentum_initialization():
    model = create_model()
    linear = get_linear(model)

    optimizer = Momentum(
        model,
        lr=0.01,
        momentum=0.9,
    )

    assert optimizer.lr == 0.01
    assert optimizer.momentum == 0.9

    # LinearのW, bの2パラメータ
    assert len(optimizer.v) == 2

    assert optimizer.v[0].shape == linear.W.shape
    assert optimizer.v[1].shape == linear.b.shape

    np.testing.assert_array_equal(
        optimizer.v[0],
        np.zeros_like(linear.W),
    )

    np.testing.assert_array_equal(
        optimizer.v[1],
        np.zeros_like(linear.b),
    )


def test_momentum_first_update():
    """
    初回はvelocity=0なので、

        v = grad
        param = param - lr * grad

    となることを確認する。
    """
    model = create_model()
    linear = get_linear(model)

    optimizer = Momentum(
        model,
        lr=0.1,
        momentum=0.9,
    )

    linear.dW = np.array([
        [1.0, -2.0, 3.0],
        [4.0, 5.0, -6.0],
    ])

    linear.db = np.array([
        0.5,
        -1.0,
        2.0,
    ])

    old_W = linear.W.copy()
    old_b = linear.b.copy()

    expected_v_W = linear.dW.copy()
    expected_v_b = linear.db.copy()

    expected_W = (
        old_W
        - optimizer.lr * expected_v_W
    )

    expected_b = (
        old_b
        - optimizer.lr * expected_v_b
    )

    # update()だけ呼ぶことで、
    # zero_gradの影響を受けず更新式そのものを検証する
    optimizer.update()

    np.testing.assert_allclose(
        optimizer.v[0],
        expected_v_W,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        optimizer.v[1],
        expected_v_b,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear.W,
        expected_W,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear.b,
        expected_b,
        rtol=1e-7,
        atol=1e-7,
    )


def test_momentum_second_update_accumulates_velocity():
    """
    2step目で

        v2 = momentum * v1 + grad2

    が正しく計算されることを確認する。
    """
    model = create_model()
    linear = get_linear(model)

    lr = 0.1
    momentum = 0.9

    optimizer = Momentum(
        model,
        lr=lr,
        momentum=momentum,
    )

    grad1_W = np.array([
        [1.0, -2.0, 3.0],
        [4.0, 5.0, -6.0],
    ])

    grad1_b = np.array([
        0.5,
        -1.0,
        2.0,
    ])

    grad2_W = np.array([
        [-0.5, 1.0, 2.0],
        [3.0, -4.0, 0.5],
    ])

    grad2_b = np.array([
        1.0,
        0.5,
        -2.0,
    ])

    initial_W = linear.W.copy()
    initial_b = linear.b.copy()

    # -------------------------
    # 1 step目
    # -------------------------

    linear.dW = grad1_W.copy()
    linear.db = grad1_b.copy()

    optimizer.update()

    expected_v1_W = grad1_W
    expected_v1_b = grad1_b

    expected_W1 = (
        initial_W
        - lr * expected_v1_W
    )

    expected_b1 = (
        initial_b
        - lr * expected_v1_b
    )

    np.testing.assert_allclose(
        linear.W,
        expected_W1,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear.b,
        expected_b1,
        rtol=1e-7,
        atol=1e-7,
    )

    # -------------------------
    # 2 step目
    # -------------------------

    linear.dW = grad2_W.copy()
    linear.db = grad2_b.copy()

    optimizer.update()

    expected_v2_W = (
        momentum * expected_v1_W
        + grad2_W
    )

    expected_v2_b = (
        momentum * expected_v1_b
        + grad2_b
    )

    expected_W2 = (
        expected_W1
        - lr * expected_v2_W
    )

    expected_b2 = (
        expected_b1
        - lr * expected_v2_b
    )

    np.testing.assert_allclose(
        optimizer.v[0],
        expected_v2_W,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        optimizer.v[1],
        expected_v2_b,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear.W,
        expected_W2,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear.b,
        expected_b2,
        rtol=1e-7,
        atol=1e-7,
    )


def test_momentum_step_zeroes_gradients():
    model = create_model()
    linear = get_linear(model)

    optimizer = Momentum(
        model,
        lr=0.1,
        momentum=0.9,
    )

    linear.dW = np.array([
        [1.0, -2.0, 3.0],
        [4.0, 5.0, -6.0],
    ])

    linear.db = np.array([
        0.5,
        -1.0,
        2.0,
    ])

    optimizer.step()

    np.testing.assert_array_equal(
        linear.dW,
        np.zeros_like(linear.dW),
    )

    np.testing.assert_array_equal(
        linear.db,
        np.zeros_like(linear.db),
    )


def test_momentum_zero_grad_does_not_change_parameters():
    model = create_model()
    linear = get_linear(model)

    optimizer = Momentum(
        model,
        lr=0.1,
        momentum=0.9,
    )

    linear.dW = np.ones_like(linear.W)
    linear.db = np.ones_like(linear.b)

    old_W = linear.W.copy()
    old_b = linear.b.copy()

    optimizer.zero_grad()

    np.testing.assert_array_equal(
        linear.W,
        old_W,
    )

    np.testing.assert_array_equal(
        linear.b,
        old_b,
    )

    np.testing.assert_array_equal(
        linear.dW,
        np.zeros_like(linear.dW),
    )

    np.testing.assert_array_equal(
        linear.db,
        np.zeros_like(linear.db),
    )


def test_momentum_updates_multiple_layers():
    model = Sequential([
        Linear(
            shape=(2, 3),
            args=LINEAR_ARGS,
        ),
        Linear(
            shape=(3, 2),
            args=LINEAR_ARGS,
        ),
    ])

    linear1 = model.layers[0]
    linear2 = model.layers[1]

    assert isinstance(linear1, Linear)
    assert isinstance(linear2, Linear)

    optimizer = Momentum(
        model,
        lr=0.1,
        momentum=0.9,
    )

    linear1.dW = np.ones_like(linear1.W)
    linear1.db = np.ones_like(linear1.b)

    linear2.dW = (
        np.ones_like(linear2.W) * 2.0
    )
    linear2.db = (
        np.ones_like(linear2.b) * 2.0
    )

    old_W1 = linear1.W.copy()
    old_b1 = linear1.b.copy()

    old_W2 = linear2.W.copy()
    old_b2 = linear2.b.copy()

    optimizer.step()

    np.testing.assert_allclose(
        linear1.W,
        old_W1 - 0.1,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear1.b,
        old_b1 - 0.1,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear2.W,
        old_W2 - 0.2,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear2.b,
        old_b2 - 0.2,
        rtol=1e-7,
        atol=1e-7,
    )


def test_momentum_against_pytorch_multiple_steps():
    """
    複数stepにわたりtorch.optim.SGD(momentum=...)
    とパラメータ更新が一致することを確認する。
    """
    model = create_model()
    linear = get_linear(model)

    lr = 0.03
    momentum = 0.8

    optimizer = Momentum(
        model,
        lr=lr,
        momentum=momentum,
    )

    # PyTorch側を完全に同じ初期値にする
    W_torch = torch.tensor(
        linear.W.copy(),
        dtype=torch.float64,
        requires_grad=True,
    )

    b_torch = torch.tensor(
        linear.b.copy(),
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_optimizer = torch.optim.SGD(
        [W_torch, b_torch],
        lr=lr,
        momentum=momentum,
    )

    gradients = [
        (
            np.array([
                [1.0, -2.0, 3.0],
                [4.0, 5.0, -6.0],
            ]),
            np.array([
                0.5,
                -1.0,
                2.0,
            ]),
        ),
        (
            np.array([
                [-0.5, 1.0, 2.0],
                [3.0, -4.0, 0.5],
            ]),
            np.array([
                1.0,
                0.5,
                -2.0,
            ]),
        ),
        (
            np.array([
                [2.0, 0.5, -1.0],
                [-2.0, 1.5, 3.0],
            ]),
            np.array([
                -0.5,
                2.0,
                1.0,
            ]),
        ),
    ]

    for grad_W, grad_b in gradients:
        # -------------------------
        # 自作Momentum
        # -------------------------

        linear.dW = grad_W.copy()
        linear.db = grad_b.copy()

        optimizer.step()

        # -------------------------
        # PyTorch
        # -------------------------

        W_torch.grad = torch.tensor(
            grad_W,
            dtype=torch.float64,
        )

        b_torch.grad = torch.tensor(
            grad_b,
            dtype=torch.float64,
        )

        torch_optimizer.step()

        # -------------------------
        # 比較
        # -------------------------

        np.testing.assert_allclose(
            linear.W,
            W_torch.detach().numpy(),
            rtol=1e-7,
            atol=1e-7,
        )

        np.testing.assert_allclose(
            linear.b,
            b_torch.detach().numpy(),
            rtol=1e-7,
            atol=1e-7,
        )

        # 自作step()では勾配がzero化される
        np.testing.assert_array_equal(
            linear.dW,
            np.zeros_like(linear.dW),
        )

        np.testing.assert_array_equal(
            linear.db,
            np.zeros_like(linear.db),
        )


def test_momentum_zero_learning_rate_does_not_change_parameters():
    """
    lr=0ならvelocityは更新されるが、
    parameterそのものは変化しない。
    """
    model = create_model()
    linear = get_linear(model)

    optimizer = Momentum(
        model,
        lr=0.0,
        momentum=0.9,
    )

    linear.dW = np.ones_like(linear.W)
    linear.db = np.ones_like(linear.b)

    old_W = linear.W.copy()
    old_b = linear.b.copy()

    optimizer.step()

    np.testing.assert_array_equal(
        linear.W,
        old_W,
    )

    np.testing.assert_array_equal(
        linear.b,
        old_b,
    )

    # lr=0でもvelocity自体は更新される
    np.testing.assert_array_equal(
        optimizer.v[0],
        np.ones_like(linear.W),
    )

    np.testing.assert_array_equal(
        optimizer.v[1],
        np.ones_like(linear.b),
    )