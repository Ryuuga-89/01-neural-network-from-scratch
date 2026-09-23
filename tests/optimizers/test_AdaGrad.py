import numpy as np
import torch

from src.neural_networks.layers.basic_layers import Linear
from src.neural_networks.layers.other_layers import Sequential
from src.neural_networks.optimizers.optimizers import AdaGrad


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


def test_adagrad_initialization():
    model = create_model()
    linear = get_linear(model)

    optimizer = AdaGrad(
        model,
        lr=0.01,
        eps=1e-8,
    )

    assert optimizer.lr == 0.01
    assert optimizer.eps == 1e-8

    # LinearのW, bに対応
    assert len(optimizer.h) == 2

    assert optimizer.h[0].shape == linear.W.shape
    assert optimizer.h[1].shape == linear.b.shape

    np.testing.assert_array_equal(
        optimizer.h[0],
        np.zeros_like(linear.W),
    )

    np.testing.assert_array_equal(
        optimizer.h[1],
        np.zeros_like(linear.b),
    )


def test_adagrad_first_update():
    model = create_model()
    linear = get_linear(model)

    lr = 0.1
    eps = 1e-8

    optimizer = AdaGrad(
        model,
        lr=lr,
        eps=eps,
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

    expected_h_W = linear.dW ** 2
    expected_h_b = linear.db ** 2

    expected_W = (
        old_W
        - (
            lr
            / (np.sqrt(expected_h_W) + eps)
        )
        * linear.dW
    )

    expected_b = (
        old_b
        - (
            lr
            / (np.sqrt(expected_h_b) + eps)
        )
        * linear.db
    )

    optimizer.update()

    np.testing.assert_allclose(
        optimizer.h[0],
        expected_h_W,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        optimizer.h[1],
        expected_h_b,
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


def test_adagrad_second_update_accumulates_squared_gradients():
    model = create_model()
    linear = get_linear(model)

    lr = 0.1
    eps = 1e-8

    optimizer = AdaGrad(
        model,
        lr=lr,
        eps=eps,
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

    expected_h1_W = grad1_W ** 2
    expected_h1_b = grad1_b ** 2

    expected_W1 = (
        initial_W
        - (
            lr
            / (np.sqrt(expected_h1_W) + eps)
        )
        * grad1_W
    )

    expected_b1 = (
        initial_b
        - (
            lr
            / (np.sqrt(expected_h1_b) + eps)
        )
        * grad1_b
    )

    # -------------------------
    # 2 step目
    # -------------------------

    linear.dW = grad2_W.copy()
    linear.db = grad2_b.copy()

    optimizer.update()

    expected_h2_W = (
        expected_h1_W
        + grad2_W ** 2
    )

    expected_h2_b = (
        expected_h1_b
        + grad2_b ** 2
    )

    expected_W2 = (
        expected_W1
        - (
            lr
            / (np.sqrt(expected_h2_W) + eps)
        )
        * grad2_W
    )

    expected_b2 = (
        expected_b1
        - (
            lr
            / (np.sqrt(expected_h2_b) + eps)
        )
        * grad2_b
    )

    np.testing.assert_allclose(
        optimizer.h[0],
        expected_h2_W,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        optimizer.h[1],
        expected_h2_b,
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


def test_adagrad_same_gradient_has_smaller_second_update():
    """
    同じ勾配を繰り返し与えると、
    hが蓄積するため2step目の更新量は小さくなる。
    """
    model = create_model()
    linear = get_linear(model)

    optimizer = AdaGrad(
        model,
        lr=0.1,
        eps=1e-8,
    )

    grad_W = np.ones_like(linear.W)
    grad_b = np.ones_like(linear.b)

    initial_W = linear.W.copy()
    initial_b = linear.b.copy()

    # 1 step目
    linear.dW = grad_W.copy()
    linear.db = grad_b.copy()

    optimizer.update()

    W_after_1 = linear.W.copy()
    b_after_1 = linear.b.copy()

    update1_W = np.abs(
        W_after_1 - initial_W
    )

    update1_b = np.abs(
        b_after_1 - initial_b
    )

    # 2 step目
    linear.dW = grad_W.copy()
    linear.db = grad_b.copy()

    optimizer.update()

    W_after_2 = linear.W.copy()
    b_after_2 = linear.b.copy()

    update2_W = np.abs(
        W_after_2 - W_after_1
    )

    update2_b = np.abs(
        b_after_2 - b_after_1
    )

    assert np.all(
        update2_W < update1_W
    )

    assert np.all(
        update2_b < update1_b
    )


def test_adagrad_step_zeroes_gradients():
    model = create_model()
    linear = get_linear(model)

    optimizer = AdaGrad(
        model,
        lr=0.1,
        eps=1e-8,
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


def test_adagrad_updates_multiple_layers():
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

    optimizer = AdaGrad(
        model,
        lr=0.1,
        eps=1e-8,
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

    # 1step目なので
    # grad / sqrt(grad^2) ≈ sign(grad)
    expected_W1 = (
        old_W1
        - (
            0.1
            / (
                np.sqrt(
                    np.ones_like(linear1.W)
                )
                + 1e-8
            )
        )
        * np.ones_like(linear1.W)
    )

    expected_b1 = (
        old_b1
        - (
            0.1
            / (
                np.sqrt(
                    np.ones_like(linear1.b)
                )
                + 1e-8
            )
        )
        * np.ones_like(linear1.b)
    )

    expected_W2 = (
        old_W2
        - (
            0.1
            / (
                np.sqrt(
                    np.ones_like(linear2.W) * 4.0
                )
                + 1e-8
            )
        )
        * (
            np.ones_like(linear2.W) * 2.0
        )
    )

    expected_b2 = (
        old_b2
        - (
            0.1
            / (
                np.sqrt(
                    np.ones_like(linear2.b) * 4.0
                )
                + 1e-8
            )
        )
        * (
            np.ones_like(linear2.b) * 2.0
        )
    )

    np.testing.assert_allclose(
        linear1.W,
        expected_W1,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear1.b,
        expected_b1,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear2.W,
        expected_W2,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear2.b,
        expected_b2,
        rtol=1e-7,
        atol=1e-7,
    )


def test_adagrad_against_pytorch_multiple_steps():
    model = create_model()
    linear = get_linear(model)

    lr = 0.03
    eps = 1e-8

    optimizer = AdaGrad(
        model,
        lr=lr,
        eps=eps,
    )

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

    torch_optimizer = torch.optim.Adagrad(
        [W_torch, b_torch],
        lr=lr,
        eps=eps,
        lr_decay=0.0,
        weight_decay=0.0,
        initial_accumulator_value=0.0,
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
        # 自作AdaGrad
        linear.dW = grad_W.copy()
        linear.db = grad_b.copy()

        optimizer.step()

        # PyTorch
        W_torch.grad = torch.tensor(
            grad_W,
            dtype=torch.float64,
        )

        b_torch.grad = torch.tensor(
            grad_b,
            dtype=torch.float64,
        )

        torch_optimizer.step()

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

        np.testing.assert_array_equal(
            linear.dW,
            np.zeros_like(linear.dW),
        )

        np.testing.assert_array_equal(
            linear.db,
            np.zeros_like(linear.db),
        )


def test_adagrad_zero_gradient_does_not_change_parameter():
    model = create_model()
    linear = get_linear(model)

    optimizer = AdaGrad(
        model,
        lr=0.1,
        eps=1e-8,
    )

    linear.dW = np.zeros_like(linear.W)
    linear.db = np.zeros_like(linear.b)

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

    np.testing.assert_array_equal(
        optimizer.h[0],
        np.zeros_like(linear.W),
    )

    np.testing.assert_array_equal(
        optimizer.h[1],
        np.zeros_like(linear.b),
    )


def test_adagrad_zero_learning_rate_does_not_change_parameters():
    model = create_model()
    linear = get_linear(model)

    optimizer = AdaGrad(
        model,
        lr=0.0,
        eps=1e-8,
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

    # lr=0でも累積二乗和そのものは更新される
    np.testing.assert_array_equal(
        optimizer.h[0],
        np.ones_like(linear.W),
    )

    np.testing.assert_array_equal(
        optimizer.h[1],
        np.ones_like(linear.b),
    )