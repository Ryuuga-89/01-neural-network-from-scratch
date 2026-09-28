"""
CNNの多クラス分類が正常に動作するかの実験。

sklearn Digitsデータセットから各クラス5件、合計50件を抽出し、
自作Conv2D / MaxPooling2Dを含むCNNがoverfitできるかを確認する。

最適化器としてAdamを使用する。
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from src.neural_networks.data_loader.data_loader import DataLoader
from src.neural_networks.layers.basic_layers import Flatten, Linear, ReLU
from src.neural_networks.layers.convolutional_layers import Conv2D, MaxPooling2D
from src.neural_networks.layers.other_layers import Sequential
from src.neural_networks.loss_functions.loss_functions import CrossEntropyLoss
from src.neural_networks.optimizers.optimizers import Adam
from src.neural_networks.trainer.trainer import Trainer

SEED = 42


LINEAR_ARGS_HE = {
    "weight_init_method": {
        "method_name": "He",
        "distribution": "normal",
        "seed": SEED,
    },

    "bias_init_method": {
        "method_name": "Zeros",
    },
}


CONV_ARGS_HE = {
    "weight_init_method": {
        "method_name": "He",
        "distribution": "normal",
        "seed": SEED,
    },

    "bias_init_method": {
        "method_name": "Zeros",
    },

    "Conv2D": {
        "stride": 1,
        "padding_mode": "Zeros",
        "padding_length": 1,
    },
}


MAX_POOL_ARGS = {
    "MaxPooling2D": {
        "stride": 2,
        "padding_mode": "Zeros",
        "padding_length": 0,
    }
}

# accuracy評価用
def multiclass_accuracy(model, data_loader):
    model.train = False

    correct = 0
    total = 0

    for x_batch, t_batch in data_loader:
        out = model.forward_propagation(x_batch)

        pred = out.argmax(axis=1)

        correct += (pred == t_batch).sum()
        total += len(t_batch)

    return correct / total


def main():
    # sklearnはデータ準備のみに使用
    X, y = load_digits(return_X_y=True)

    # [N, 64] -> [N, 1, 8, 8]
    X = X.reshape(-1, 1, 8, 8)

    # Digitsのpixel valueは0〜16
    X = X.astype(np.float64) / 16.0

    # 各クラス5件 = 合計50件だけ使用
    X_tiny, _, y_tiny, _ = train_test_split(
        X,
        y,
        train_size=50,
        random_state=SEED,
        stratify=y,
    )

    # 念のため各クラス5件ずつ抽出されていることを確認
    classes, counts = np.unique(
        y_tiny,
        return_counts=True,
    )

    print(
        "Class distribution:",
        dict(zip(classes, counts)),
    )

    # ここから自作コンポーネントのみ
    train_loader = DataLoader(
        X_tiny,
        y_tiny,
        batch_size=len(X_tiny),
        shuffle=True,
        drop_last=False,
        seed=SEED,
    )

    model = Sequential(
        [
            # [B, 1, 8, 8]
            Conv2D(
                shape=(4, 1, 3, 3),
                args=CONV_ARGS_HE,
            ),

            # [B, 4, 8, 8]
            ReLU(),

            MaxPooling2D(
                shape=(2, 2),
                args=MAX_POOL_ARGS,
            ),

            # [B, 4, 4, 4]
            Conv2D(
                shape=(8, 4, 3, 3),
                args=CONV_ARGS_HE,
            ),

            # [B, 8, 4, 4]
            ReLU(),

            MaxPooling2D(
                shape=(2, 2),
                args=MAX_POOL_ARGS,
            ),

            # [B, 8, 2, 2]
            Flatten(),

            # 8 * 2 * 2 = 32
            Linear(
                shape=(32, 32),
                args=LINEAR_ARGS_HE,
            ),

            ReLU(),

            Linear(
                shape=(32, 10),
                args=LINEAR_ARGS_HE,
            ),
        ]
    )

    loss_function = CrossEntropyLoss()

    optimizer = Adam(model)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        train_loader=train_loader,
    )

    trainer.fit(1000)

    accuracy = multiclass_accuracy(
        model,
        train_loader,
    )

    print(
        f"Train Accuracy: {accuracy:.4f}"
    )


if __name__ == "__main__":
    main()