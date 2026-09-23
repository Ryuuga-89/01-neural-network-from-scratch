import numpy as np

from src.neural_networks.data_loader.data_loader import DataLoader
from src.neural_networks.layers.basic_layers import Linear, ReLU
from src.neural_networks.layers.other_layers import Sequential
from src.neural_networks.loss_functions.loss_functions import CrossEntropyLoss
from src.neural_networks.optimizers.optimizers import SGD
from src.neural_networks.trainer.trainer import Trainer

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
        Linear(shape=(2, 4), args=LINEAR_ARGS),
        ReLU(),
        Linear(shape=(4, 2), args=LINEAR_ARGS),
    ])


def create_dataset():
    X = np.array([
        [2.0, 1.0],
        [1.5, 0.5],
        [1.0, 1.5],
        [0.5, 1.0],
        [-1.0, -1.0],
        [-1.5, -0.5],
        [-2.0, -1.0],
        [-0.5, -1.5],
    ])

    t = np.array([
        0, 0, 0, 0,
        1, 1, 1, 1,
    ])

    return X, t


def test_train_epoch_updates_parameters():
    X, t = create_dataset()

    train_loader = DataLoader(
        X,
        t,
        batch_size=4,
        shuffle=False,
    )

    model = create_model()

    linear1 = model.layers[0]
    linear2 = model.layers[2]

    assert isinstance(linear1, Linear)
    assert isinstance(linear2, Linear)

    old_W1 = linear1.W.copy()
    old_b1 = linear1.b.copy()
    old_W2 = linear2.W.copy()
    old_b2 = linear2.b.copy()

    optimizer = SGD(model, lr=0.01)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=CrossEntropyLoss(),
        train_loader=train_loader,
    )

    loss = trainer.train_epoch()

    assert np.isfinite(loss)

    assert not np.array_equal(linear1.W, old_W1)
    assert not np.array_equal(linear1.b, old_b1)
    assert not np.array_equal(linear2.W, old_W2)
    assert not np.array_equal(linear2.b, old_b2)
    
    
def test_train_epoch_against_manual_loop():
    X, t = create_dataset()

    loader1 = DataLoader(
        X,
        t,
        batch_size=4,
        shuffle=False,
    )

    loader2 = DataLoader(
        X,
        t,
        batch_size=4,
        shuffle=False,
    )

    model1 = create_model()
    model2 = create_model()

    optimizer1 = SGD(model1, lr=0.01)
    optimizer2 = SGD(model2, lr=0.01)

    loss_function1 = CrossEntropyLoss()
    loss_function2 = CrossEntropyLoss()

    trainer = Trainer(
        model=model1,
        optimizer=optimizer1,
        loss_function=loss_function1,
        train_loader=loader1,
    )

    trainer_loss = trainer.train_epoch()

    # 手動学習ループ
    total_loss = 0.0

    for x_batch, t_batch in loader2:
        out = model2.forward_propagation(x_batch)

        loss = loss_function2.forward_propagation(
            out,
            t_batch,
        )
        total_loss += loss

        dout = loss_function2.backward_propagation(
            dout=1.0,
        )

        model2.backward_propagation(dout)
        optimizer2.step()

    expected_loss = total_loss / len(loader2)

    np.testing.assert_allclose(
        trainer_loss,
        expected_loss,
        rtol=1e-7,
        atol=1e-7,
    )

    linear1_a = model1.layers[0]
    linear2_a = model1.layers[2]

    linear1_b = model2.layers[0]
    linear2_b = model2.layers[2]

    assert isinstance(linear1_a, Linear)
    assert isinstance(linear2_a, Linear)
    assert isinstance(linear1_b, Linear)
    assert isinstance(linear2_b, Linear)

    np.testing.assert_allclose(
        linear1_a.W,
        linear1_b.W,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear1_a.b,
        linear1_b.b,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear2_a.W,
        linear2_b.W,
        rtol=1e-7,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        linear2_a.b,
        linear2_b.b,
        rtol=1e-7,
        atol=1e-7,
    )
    
    
def test_evaluate_does_not_update_parameters():
    X, t = create_dataset()

    train_loader = DataLoader(
        X,
        t,
        batch_size=4,
        shuffle=False,
    )

    val_loader = DataLoader(
        X,
        t,
        batch_size=4,
        shuffle=False,
    )

    model = create_model()

    linear1 = model.layers[0]
    linear2 = model.layers[2]

    assert isinstance(linear1, Linear)
    assert isinstance(linear2, Linear)

    optimizer = SGD(model, lr=0.01)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=CrossEntropyLoss(),
        train_loader=train_loader,
        val_loader=val_loader,
    )

    old_W1 = linear1.W.copy()
    old_b1 = linear1.b.copy()
    old_W2 = linear2.W.copy()
    old_b2 = linear2.b.copy()

    loss = trainer.evaluate()

    assert np.isfinite(loss)

    np.testing.assert_array_equal(
        linear1.W,
        old_W1,
    )
    np.testing.assert_array_equal(
        linear1.b,
        old_b1,
    )
    np.testing.assert_array_equal(
        linear2.W,
        old_W2,
    )
    np.testing.assert_array_equal(
        linear2.b,
        old_b2,
    )
    
    
def test_trainer_switches_train_mode():
    X, t = create_dataset()

    train_loader = DataLoader(
        X,
        t,
        batch_size=4,
        shuffle=False,
    )

    val_loader = DataLoader(
        X,
        t,
        batch_size=4,
        shuffle=False,
    )

    model = create_model()

    trainer = Trainer(
        model=model,
        optimizer=SGD(model, lr=0.01),
        loss_function=CrossEntropyLoss(),
        train_loader=train_loader,
        val_loader=val_loader,
    )

    trainer.evaluate()
    assert model.train is False

    trainer.train_epoch()
    assert model.train is True
    
    
def test_evaluate_without_validation_loader():
    X, t = create_dataset()

    train_loader = DataLoader(
        X,
        t,
        batch_size=4,
        shuffle=False,
    )

    model = create_model()

    trainer = Trainer(
        model=model,
        optimizer=SGD(model, lr=0.01),
        loss_function=CrossEntropyLoss(),
        train_loader=train_loader,
        val_loader=None,
    )

    assert trainer.evaluate() == 0.0
    
    
def test_fit_history():
    X, t = create_dataset()

    train_loader = DataLoader(
        X,
        t,
        batch_size=4,
        shuffle=False,
    )

    val_loader = DataLoader(
        X,
        t,
        batch_size=4,
        shuffle=False,
    )

    model = create_model()

    trainer = Trainer(
        model=model,
        optimizer=SGD(model, lr=0.01),
        loss_function=CrossEntropyLoss(),
        train_loader=train_loader,
        val_loader=val_loader,
    )

    history = trainer.fit(epochs=3)

    assert len(history["train_loss"]) == 3
    assert len(history["val_loss"]) == 3

    assert all(
        np.isfinite(loss)
        for loss in history["train_loss"]
    )

    assert all(
        np.isfinite(loss)
        for loss in history["val_loss"]
    )
    
    
def test_fit_history_without_validation():
    X, t = create_dataset()

    train_loader = DataLoader(
        X,
        t,
        batch_size=4,
        shuffle=False,
    )

    model = create_model()

    trainer = Trainer(
        model=model,
        optimizer=SGD(model, lr=0.01),
        loss_function=CrossEntropyLoss(),
        train_loader=train_loader,
    )

    history = trainer.fit(epochs=3)

    assert len(history["train_loss"]) == 3
    assert history["val_loss"] == []
    
    
def test_evaluate_uses_sample_weighted_mean():
    X, t = create_dataset()

    # 8 samplesを3, 3, 2に分割
    val_loader = DataLoader(
        X,
        t,
        batch_size=3,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        X,
        t,
        batch_size=3,
        shuffle=False,
    )

    model = create_model()
    loss_function = CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=SGD(model, lr=0.01),
        loss_function=loss_function,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    actual = trainer.evaluate()

    # 全8サンプルを一度に評価
    out = model.forward_propagation(X)
    expected = CrossEntropyLoss().forward_propagation(
        out,
        t,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-7,
        atol=1e-7,
    )