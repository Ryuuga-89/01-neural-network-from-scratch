import numpy as np

from src.neural_networks.layers.other_layers import Dropout


def create_args(
    dropout_rate: float = 0.5,
    seed: int = 42,
) -> dict:
    return {
        "Dropout": {
            "dropout_rate": dropout_rate,
            "seed": seed,
        }
    }


def test_dropout_initialization():
    dropout = Dropout(
        args=create_args(
            dropout_rate=0.3,
            seed=123,
        )
    )

    assert dropout.dropout_rate == 0.3
    assert dropout.seed == 123
    assert isinstance(
        dropout.rng,
        np.random.Generator,
    )


def test_dropout_forward_known_mask():
    """
    seedから生成される乱数を独立なGeneratorでも生成し、
    forwardが期待通りのmaskを使用していることを確認する。
    """
    dropout_rate = 0.5
    seed = 42

    dropout = Dropout(
        args=create_args(
            dropout_rate=dropout_rate,
            seed=seed,
        )
    )

    x = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    actual = dropout.forward_propagation(x)

    keep_p = 1.0 - dropout_rate

    rng = np.random.default_rng(seed)

    expected_mask = (
        rng.random(x.shape) < keep_p
    ) / keep_p

    expected = x * expected_mask

    np.testing.assert_array_equal(
        dropout.mask,
        expected_mask,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    

def test_dropout_backward_uses_forward_mask():
    dropout = Dropout(
        args=create_args(
            dropout_rate=0.5,
            seed=42,
        )
    )

    x = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    dropout.forward_propagation(x)

    dout = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ])

    actual = dropout.backward_propagation(
        dout
    )

    expected = dout * dropout.mask

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )
    

def test_dropout_eval_forward_is_identity():
    dropout = Dropout(
        args=create_args(
            dropout_rate=0.8,
            seed=42,
        )
    )

    dropout.train = False

    x = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    actual = dropout.forward_propagation(x)

    np.testing.assert_array_equal(
        actual,
        x,
    )
    
def test_dropout_eval_backward_is_identity():
    dropout = Dropout(
        args=create_args(
            dropout_rate=0.8,
            seed=42,
        )
    )

    dropout.train = False

    dout = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
    ])

    actual = dropout.backward_propagation(
        dout
    )

    np.testing.assert_array_equal(
        actual,
        dout,
    )
    
def test_dropout_same_seed_same_first_mask():
    x = np.ones((10, 10))

    dropout1 = Dropout(
        args=create_args(
            dropout_rate=0.4,
            seed=42,
        )
    )

    dropout2 = Dropout(
        args=create_args(
            dropout_rate=0.4,
            seed=42,
        )
    )

    out1 = dropout1.forward_propagation(x)
    out2 = dropout2.forward_propagation(x)

    np.testing.assert_array_equal(
        dropout1.mask,
        dropout2.mask,
    )

    np.testing.assert_array_equal(
        out1,
        out2,
    )
    
def test_dropout_generates_new_mask_each_forward():
    dropout = Dropout(
        args=create_args(
            dropout_rate=0.5,
            seed=42,
        )
    )

    x = np.ones((100, 100))

    dropout.forward_propagation(x)
    mask1 = dropout.mask.copy()

    dropout.forward_propagation(x)
    mask2 = dropout.mask.copy()

    assert not np.array_equal(
        mask1,
        mask2,
    )
    
    
def test_dropout_rate_statistically():
    dropout_rate = 0.3

    dropout = Dropout(
        args=create_args(
            dropout_rate=dropout_rate,
            seed=42,
        )
    )

    x = np.ones(100_000)

    dropout.forward_propagation(x)

    dropped_ratio = np.mean(
        dropout.mask == 0
    )

    assert np.isclose(
        dropped_ratio,
        dropout_rate,
        atol=0.01,
    )
    
def test_dropout_preserves_expected_value():
    dropout = Dropout(
        args=create_args(
            dropout_rate=0.3,
            seed=42,
        )
    )

    x = np.ones(100_000)

    out = dropout.forward_propagation(x)

    assert np.isclose(
        np.mean(out),
        1.0,
        atol=0.02,
    )
    
    
def test_dropout_zero_rate_is_identity():
    dropout = Dropout(
        args=create_args(
            dropout_rate=0.0,
            seed=42,
        )
    )

    x = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    actual = dropout.forward_propagation(x)

    np.testing.assert_array_equal(
        actual,
        x,
    )

    np.testing.assert_array_equal(
        dropout.mask,
        np.ones_like(x),
    )

    dout = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
    ])

    actual_dx = dropout.backward_propagation(
        dout
    )

    np.testing.assert_array_equal(
        actual_dx,
        dout,
    )