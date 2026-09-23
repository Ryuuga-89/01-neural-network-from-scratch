import numpy as np
import pytest

from src.neural_networks.data_loader.data_loader import DataLoader


def create_dataset():
    """
    Xとtの対応関係が分かりやすいテストデータ。
    X[i] と t[i] の対応が崩れていないことも確認できる。
    """
    X = np.array([
        [0, 100],
        [1, 101],
        [2, 102],
        [3, 103],
        [4, 104],
        [5, 105],
        [6, 106],
        [7, 107],
        [8, 108],
        [9, 109],
    ])

    t = np.arange(10)

    return X, t


def test_dataloader_len_without_drop_last():
    X, t = create_dataset()

    loader = DataLoader(
        X,
        t,
        batch_size=4,
        shuffle=False,
        drop_last=False,
    )

    assert len(loader) == 3


def test_dataloader_len_with_drop_last():
    X, t = create_dataset()

    loader = DataLoader(
        X,
        t,
        batch_size=4,
        shuffle=False,
        drop_last=True,
    )

    assert len(loader) == 2
    
    
def test_dataloader_without_shuffle():
    X, t = create_dataset()

    loader = DataLoader(
        X,
        t,
        batch_size=4,
        shuffle=False,
        drop_last=False,
    )

    batches = list(loader)

    assert len(batches) == 3

    X1, t1 = batches[0]
    X2, t2 = batches[1]
    X3, t3 = batches[2]

    np.testing.assert_array_equal(
        X1,
        X[0:4],
    )
    np.testing.assert_array_equal(
        t1,
        t[0:4],
    )

    np.testing.assert_array_equal(
        X2,
        X[4:8],
    )
    np.testing.assert_array_equal(
        t2,
        t[4:8],
    )

    np.testing.assert_array_equal(
        X3,
        X[8:10],
    )
    np.testing.assert_array_equal(
        t3,
        t[8:10],
    )
    
    
def test_dataloader_drop_last():
    X, t = create_dataset()

    loader = DataLoader(
        X,
        t,
        batch_size=4,
        shuffle=False,
        drop_last=True,
    )

    batches = list(loader)

    assert len(batches) == 2

    X1, t1 = batches[0]
    X2, t2 = batches[1]

    assert X1.shape[0] == 4
    assert X2.shape[0] == 4

    np.testing.assert_array_equal(
        t1,
        np.array([0, 1, 2, 3]),
    )

    np.testing.assert_array_equal(
        t2,
        np.array([4, 5, 6, 7]),
    )
    
    
def test_dataloader_shuffle_contains_all_samples():
    X, t = create_dataset()

    loader = DataLoader(
        X,
        t,
        batch_size=3,
        shuffle=True,
        drop_last=False,
        seed=42,
    )

    collected_t = []

    for _, batch_t in loader:
        collected_t.extend(batch_t.tolist())

    assert len(collected_t) == 10

    np.testing.assert_array_equal(
        np.sort(collected_t),
        np.arange(10),
    )
    
    
def test_dataloader_preserves_x_target_pair():
    X, t = create_dataset()

    loader = DataLoader(
        X,
        t,
        batch_size=3,
        shuffle=True,
        seed=42,
    )

    for batch_X, batch_t in loader:
        np.testing.assert_array_equal(
            batch_X[:, 0],
            batch_t,
        )
        
        
def test_dataloader_same_seed():
    X, t = create_dataset()

    loader1 = DataLoader(
        X,
        t,
        batch_size=3,
        shuffle=True,
        seed=42,
    )

    loader2 = DataLoader(
        X,
        t,
        batch_size=3,
        shuffle=True,
        seed=42,
    )

    order1 = np.concatenate([
        batch_t
        for _, batch_t in loader1
    ])

    order2 = np.concatenate([
        batch_t
        for _, batch_t in loader2
    ])

    np.testing.assert_array_equal(
        order1,
        order2,
    )
    
    
def test_dataloader_reshuffles_each_epoch():
    X, t = create_dataset()

    loader = DataLoader(
        X,
        t,
        batch_size=3,
        shuffle=True,
        seed=42,
    )

    order_epoch1 = np.concatenate([
        batch_t
        for _, batch_t in loader
    ])

    order_epoch2 = np.concatenate([
        batch_t
        for _, batch_t in loader
    ])

    assert not np.array_equal(
        order_epoch1,
        order_epoch2,
    )
    
    
def test_dataloader_no_shuffle_same_each_epoch():
    X, t = create_dataset()

    loader = DataLoader(
        X,
        t,
        batch_size=3,
        shuffle=False,
    )

    order_epoch1 = np.concatenate([
        batch_t
        for _, batch_t in loader
    ])

    order_epoch2 = np.concatenate([
        batch_t
        for _, batch_t in loader
    ])

    np.testing.assert_array_equal(
        order_epoch1,
        order_epoch2,
    )
    

def test_dataloader_mismatched_sample_count():
    X = np.zeros((10, 2))
    t = np.zeros(9)

    with pytest.raises(
        ValueError,
        match="特徴量と正解データのサンプル数が一致しません",
    ):
        DataLoader(X, t)


def test_dataloader_invalid_batch_size():
    X, t = create_dataset()

    with pytest.raises(ValueError):
        DataLoader(
            X,
            t,
            batch_size=0,
        )

    with pytest.raises(ValueError):
        DataLoader(
            X,
            t,
            batch_size=-1,
        )
        
        
def test_dataloader_batch_size_larger_than_dataset():
    X, t = create_dataset()

    loader = DataLoader(
        X,
        t,
        batch_size=100,
        shuffle=False,
        drop_last=False,
    )

    batches = list(loader)

    assert len(batches) == 1

    batch_X, batch_t = batches[0]

    np.testing.assert_array_equal(
        batch_X,
        X,
    )

    np.testing.assert_array_equal(
        batch_t,
        t,
    )
    
    
def test_dataloader_batch_size_larger_than_dataset_drop_last():
    X, t = create_dataset()

    loader = DataLoader(
        X,
        t,
        batch_size=100,
        shuffle=False,
        drop_last=True,
    )

    assert len(loader) == 0
    assert list(loader) == []