"""
MLPの二値分類が正常に動作するかの実験
BreastCancerデータの中から少数を抽出し、overfitできるかを調べる。

同時に最適化器としてAdaGradをテストする。
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.neural_networks.data_loader.data_loader import DataLoader
from src.neural_networks.layers.basic_layers import Linear, ReLU
from src.neural_networks.layers.other_layers import Sequential
from src.neural_networks.loss_functions.loss_functions import BCEWithLogitsLoss
from src.neural_networks.optimizers.optimizers import AdaGrad
from src.neural_networks.trainer.trainer import Trainer

SEED = 42

LINEAR_ARGS_HE = {
    "weight_init_method": {
        "method_name": "He",
        "distribution": "normal",
        "seed": SEED
    },
    
    "bias_init_method": {
        "method_name": "Zeros"
    }
}


def binary_accuracy(model, data_loader):
    model.train = False

    correct = 0
    total = 0

    for x_batch, t_batch in data_loader:
        logits = model.forward_propagation(x_batch)

        # sigmoid(logit) >= 0.5 と同値
        pred = (logits >= 0).astype(int).reshape(-1)
        target = t_batch.reshape(-1)

        correct += np.sum(pred == target)
        total += len(target)

    return correct / total


def main():
    # sklearnを用いたデータ準備
    X, y = load_breast_cancer(return_X_y=True)

    # overfit確認用に20サンプルのみ使用
    X_tiny, _, y_tiny, _ = train_test_split(
        X,
        y,
        train_size=20,
        random_state=SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    X_tiny = scaler.fit_transform(X_tiny)

    # BCEWithLogitsLossのmodel outputとshapeを合わせる
    y_tiny = y_tiny.reshape(-1, 1).astype(float)

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
            Linear(shape=(30, 64), args=LINEAR_ARGS_HE),
            ReLU(),
            Linear(shape=(64, 32), args=LINEAR_ARGS_HE),
            ReLU(),
            Linear(shape=(32, 1), args=LINEAR_ARGS_HE),
        ]
    )

    loss_function = BCEWithLogitsLoss()

    optimizer = AdaGrad(model)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        train_loader=train_loader,
    )

    epochs = 1000

    trainer.fit(epochs)

    accuracy = binary_accuracy(
        model,
        train_loader,
    )

    print(
        f"Train Accuracy: "
        f"{accuracy:.4f}"
    )


if __name__ == "__main__":
    main()