import pytest

from src.neural_networks.layers.basic_layers import Linear

TEST_LINEAR_ARGS: dict = {
    "weight_init_method": {
        "method_name": "He",
        "distribution": "normal"
    },
    
    "bias_init_method": {
        "method_name": "Zeros"
    }
}

def test_Linear_initialization():
    test_linear: Linear = Linear((3, 2), TEST_LINEAR_ARGS)
    
    assert test_linear.W.shape == (3, 2)
    assert test_linear.b.shape == (2,)
    assert test_linear.n_in == 3
    assert test_linear.n_out == 2
    