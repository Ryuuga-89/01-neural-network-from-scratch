"""
MLPの回帰タスクが正常に動作するかの実験
Diabetesデータの中から少数を抽出し、overfitできるかを調べる。

最適化器としてはSGDを用いる
"""
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.neural_networks.data_loader.data_loader import DataLoader
from src.neural_networks.layers.basic_layers import Linear, ReLU
from src.neural_networks.layers.other_layers import Sequential
from src.neural_networks.loss_functions.loss_functions import MeanSquaredErrorLoss
from src.neural_networks.optimizers.optimizers import SGD
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

def main():
    # sklearnを用いたデータ準備
    X, y = load_diabetes(return_X_y=True)

    # overfit確認用に20サンプルだけ使用
    X_tiny, _, y_tiny, _ = train_test_split(
        X,
        y,
        train_size=20,
        random_state=SEED,
    )

    # 入力を標準化
    x_scaler = StandardScaler()
    X_tiny = x_scaler.fit_transform(X_tiny)

    # 回帰targetも標準化して最適化しやすくする
    y_scaler = StandardScaler()
    y_tiny = y_scaler.fit_transform(
        y_tiny.reshape(-1, 1)
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
            Linear(shape=(10, 64), args=LINEAR_ARGS_HE),
            ReLU(),
            Linear(shape=(64, 64), args=LINEAR_ARGS_HE),
            ReLU(),
            Linear(shape=(64, 1), args=LINEAR_ARGS_HE),
        ]
    )

    loss_function = MeanSquaredErrorLoss()

    optimizer = SGD(model)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        train_loader=train_loader,
    )

    epochs = 5000

    history = trainer.fit(epochs)

    final_loss = history["train_loss"][-1]

    print(f"Final Train MSE: {final_loss:.6f}")


if __name__ == "__main__":
    main()