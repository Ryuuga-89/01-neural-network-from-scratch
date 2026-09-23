import numpy as np
import torch

from src.neural_networks.layers.basic_layers import Linear
from src.neural_networks.layers.other_layers import Sequential
from src.neural_networks.optimizers.optimizers import Adam


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


def test_adam_initialization():
    model = create_model()
    linear = get_linear(model)

    optimizer = Adam(
        model,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    )

    assert optimizer.lr == 0.001
    assert optimizer.beta1 == 0.9
    assert optimizer.beta2 == 0.999
    assert optimizer.eps == 1e-8
    assert optimizer.iter == 0

    assert len(optimizer.m) == 2
    assert len(optimizer.v) == 2

    assert optimizer.m[0].shape == linear.W.shape
    assert optimizer.m[1].shape == linear.b.shape
    assert optimizer.v[0].shape == linear.W.shape
    assert optimizer.v[1].shape == linear.b.shape

    np.testing.assert_array_equal(
        optimizer.m[0],
        np.zeros_like(linear.W),
    )

    np.testing.assert_array_equal(
        optimizer.m[1],
        np.zeros_like(linear.b),
    )

    np.testing.assert_array_equal(
        optimizer.v[0],
        np.zeros_like(linear.W),
    )

    np.testing.assert_array_equal(
        optimizer.v[1],
        np.zeros_like(linear.b),
    )


def test_adam_first_update():
    """
    標準Adamの定義に従って、
    1step目のm, v, parameterを手計算と比較する。
    """
    model = create_model()
    linear = get_linear(model)

    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-2

    optimizer = Adam(
        model,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )

    grad_W = np.array([
        [1.0, -2.0, 3.0],
        [4.0, 5.0, -6.0],
    ])

    grad_b = np.array([
        0.5,
        -1.0,
        2.0,
    ])

    linear.dW = grad_W.copy()
    linear.db = grad_b.copy()

    old_W = linear.W.copy()
    old_b = linear.b.copy()

    # 1次・2次モーメント
    expected_m_W = (
        (1.0 - beta1) * grad_W
    )
    expected_m_b = (
        (1.0 - beta1) * grad_b
    )

    expected_v_W = (
        (1.0 - beta2) * grad_W ** 2
    )
    expected_v_b = (
        (1.0 - beta2) * grad_b ** 2
    )

    # bias correction
    expected_m_hat_W = (
        expected_m_W
        / (1.0 - beta1)
    )
    expected_m_hat_b = (
        expected_m_b
        / (1.0 - beta1)
    )

    expected_v_hat_W = (
        expected_v_W
        / (1.0 - beta2)
    )
    expected_v_hat_b = (
        expected_v_b
        / (1.0 - beta2)
    )

    expected_W = (
        old_W
        - lr
        * expected_m_hat_W
        / (
            np.sqrt(expected_v_hat_W)
            + eps
        )
    )

    expected_b = (
        old_b
        - lr
        * expected_m_hat_b
        / (
            np.sqrt(expected_v_hat_b)
            + eps
        )
    )

    optimizer.update()

    assert optimizer.iter == 1

    np.testing.assert_allclose(
        optimizer.m[0],
        expected_m_W,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        optimizer.m[1],
        expected_m_b,
        rtol=1e-7,
        atol=1e-7,
    )

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


def test_adam_second_update():
    """
    2step目までm, vがEMAとして蓄積され、
    bias correctionもiter=2として適用されることを確認する。
    """
    model = create_model()
    linear = get_linear(model)

    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    optimizer = Adam(
        model,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
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
    # step 1
    # -------------------------

    linear.dW = grad1_W.copy()
    linear.db = grad1_b.copy()

    optimizer.update()

    m1_W = (
        beta1 * np.zeros_like(grad1_W)
        + (1.0 - beta1) * grad1_W
    )

    m1_b = (
        beta1 * np.zeros_like(grad1_b)
        + (1.0 - beta1) * grad1_b
    )

    v1_W = (
        beta2 * np.zeros_like(grad1_W)
        + (1.0 - beta2) * grad1_W ** 2
    )

    v1_b = (
        beta2 * np.zeros_like(grad1_b)
        + (1.0 - beta2) * grad1_b ** 2
    )

    m1_hat_W = m1_W / (1.0 - beta1)
    m1_hat_b = m1_b / (1.0 - beta1)

    v1_hat_W = v1_W / (1.0 - beta2)
    v1_hat_b = v1_b / (1.0 - beta2)

    expected_W1 = (
        initial_W
        - lr
        * m1_hat_W
        / (np.sqrt(v1_hat_W) + eps)
    )

    expected_b1 = (
        initial_b
        - lr
        * m1_hat_b
        / (np.sqrt(v1_hat_b) + eps)
    )

    # -------------------------
    # step 2
    # -------------------------

    linear.dW = grad2_W.copy()
    linear.db = grad2_b.copy()

    optimizer.update()

    m2_W = (
        beta1 * m1_W
        + (1.0 - beta1) * grad2_W
    )

    m2_b = (
        beta1 * m1_b
        + (1.0 - beta1) * grad2_b
    )

    v2_W = (
        beta2 * v1_W
        + (1.0 - beta2) * grad2_W ** 2
    )

    v2_b = (
        beta2 * v1_b
        + (1.0 - beta2) * grad2_b ** 2
    )

    m2_hat_W = (
        m2_W
        / (1.0 - beta1 ** 2)
    )

    m2_hat_b = (
        m2_b
        / (1.0 - beta1 ** 2)
    )

    v2_hat_W = (
        v2_W
        / (1.0 - beta2 ** 2)
    )

    v2_hat_b = (
        v2_b
        / (1.0 - beta2 ** 2)
    )

    expected_W2 = (
        expected_W1
        - lr
        * m2_hat_W
        / (
            np.sqrt(v2_hat_W)
            + eps
        )
    )

    expected_b2 = (
        expected_b1
        - lr
        * m2_hat_b
        / (
            np.sqrt(v2_hat_b)
            + eps
        )
    )

    assert optimizer.iter == 2

    np.testing.assert_allclose(
        optimizer.m[0],
        m2_W,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        optimizer.m[1],
        m2_b,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        optimizer.v[0],
        v2_W,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        optimizer.v[1],
        v2_b,
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


def test_adam_step_increments_iteration():
    model = create_model()
    linear = get_linear(model)

    optimizer = Adam(model)

    linear.dW = np.ones_like(linear.W)
    linear.db = np.ones_like(linear.b)

    assert optimizer.iter == 0

    optimizer.step()

    assert optimizer.iter == 1

    linear.dW[:] = 1.0
    linear.db[:] = 1.0

    optimizer.step()

    assert optimizer.iter == 2


def test_adam_step_zeroes_gradients():
    model = create_model()
    linear = get_linear(model)

    optimizer = Adam(model)

    linear.dW = np.ones_like(linear.W)
    linear.db = np.ones_like(linear.b)

    optimizer.step()

    np.testing.assert_array_equal(
        linear.dW,
        np.zeros_like(linear.dW),
    )

    np.testing.assert_array_equal(
        linear.db,
        np.zeros_like(linear.db),
    )


def test_adam_updates_multiple_layers():
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

    optimizer = Adam(
        model,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
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

    assert not np.array_equal(
        linear1.W,
        old_W1,
    )

    assert not np.array_equal(
        linear1.b,
        old_b1,
    )

    assert not np.array_equal(
        linear2.W,
        old_W2,
    )

    assert not np.array_equal(
        linear2.b,
        old_b2,
    )

    np.testing.assert_array_equal(
        linear1.dW,
        np.zeros_like(linear1.dW),
    )

    np.testing.assert_array_equal(
        linear1.db,
        np.zeros_like(linear1.db),
    )

    np.testing.assert_array_equal(
        linear2.dW,
        np.zeros_like(linear2.dW),
    )

    np.testing.assert_array_equal(
        linear2.db,
        np.zeros_like(linear2.db),
    )


def test_adam_against_pytorch_multiple_steps():
    """
    標準Adamと複数stepにわたって完全一致することを確認する。

    epsを意図的に大きくすることで、
    epsの配置が誤っている実装も検出する。
    """
    model = create_model()
    linear = get_linear(model)

    lr = 0.003
    beta1 = 0.8
    beta2 = 0.95
    eps = 1e-2

    optimizer = Adam(
        model,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
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

    torch_optimizer = torch.optim.Adam(
        [W_torch, b_torch],
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=0.0,
        amsgrad=False,
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

    for step, (grad_W, grad_b) in enumerate(
        gradients,
        start=1,
    ):
        # -------------------------
        # 自作Adam
        # -------------------------

        linear.dW = grad_W.copy()
        linear.db = grad_b.copy()

        optimizer.step()

        # -------------------------
        # PyTorch Adam
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

        assert optimizer.iter == step

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


def test_adam_zero_gradient_does_not_change_parameter():
    model = create_model()
    linear = get_linear(model)

    optimizer = Adam(
        model,
        lr=0.001,
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
        optimizer.m[0],
        np.zeros_like(linear.W),
    )

    np.testing.assert_array_equal(
        optimizer.m[1],
        np.zeros_like(linear.b),
    )

    np.testing.assert_array_equal(
        optimizer.v[0],
        np.zeros_like(linear.W),
    )

    np.testing.assert_array_equal(
        optimizer.v[1],
        np.zeros_like(linear.b),
    )


def test_adam_zero_learning_rate_does_not_change_parameters():
    model = create_model()
    linear = get_linear(model)

    optimizer = Adam(
        model,
        lr=0.0,
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

    # lr=0でも内部状態は更新される
    assert optimizer.iter == 1

    assert np.any(
        optimizer.m[0] != 0
    )

    assert np.any(
        optimizer.m[1] != 0
    )

    assert np.any(
        optimizer.v[0] != 0
    )

    assert np.any(
        optimizer.v[1] != 0
    )