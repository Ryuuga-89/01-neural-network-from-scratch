"""
MLPの多クラス分類が正常に動作するかの実験
Irisデータの中から少数を抽出し、overfitできるかを調べる。

最適化器としてMomentumをテストする。
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.neural_networks.data_loader.data_loader import DataLoader
from src.neural_networks.layers.basic_layers import Linear, ReLU
from src.neural_networks.layers.other_layers import Sequential
from src.neural_networks.loss_functions.loss_functions import CrossEntropyLoss
from src.neural_networks.optimizers.optimizers import Momentum
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
    X, y = load_iris(return_X_y=True)

    # 各クラス5件 = 合計15件だけ使用
    X_tiny, _, y_tiny, _ = train_test_split(
        X,
        y,
        train_size=15,
        random_state=SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    X_tiny = scaler.fit_transform(X_tiny)

    # ここから自作コンポーネントのみ
    train_loader = DataLoader(
        X_tiny,
        y_tiny,
        batch_size=len(X_tiny),  # full batchでよい
        shuffle=True,
        drop_last=False,
        seed=SEED,
    )

    model = Sequential(
        [
            Linear(shape=(4, 32), args=LINEAR_ARGS_HE),
            ReLU(),
            Linear(shape=(32, 32), args=LINEAR_ARGS_HE),
            ReLU(),
            Linear(shape=(32, 3), args=LINEAR_ARGS_HE),
        ]
    )

    loss_function = CrossEntropyLoss()

    optimizer = Momentum(model)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        train_loader=train_loader,
    )

    trainer.fit(1000)

    accuracy = multiclass_accuracy(model, train_loader)
    
    print(f"Train Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
