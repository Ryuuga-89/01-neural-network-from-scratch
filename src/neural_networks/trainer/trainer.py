from src.neural_networks.data_loader.data_loader import DataLoader
from src.neural_networks.layers.abstract_layers import Layer
from src.neural_networks.loss_functions.loss_functions import LossFunction
from src.neural_networks.optimizers.optimizers import Optimizer


class Trainer:
    """
    学習から評価までのループをe2eで行う
    
    モデル、学習データローダー、検証データローダー、損失関数、最適化器を全てまとめ、適切な呼び出しを提供する
    """
    model: Layer
    optimizer: Optimizer
    train_loader: DataLoader
    val_loader: DataLoader | None
    loss_function: LossFunction
    history: dict[str, list[float]]
    
    def __init__(
            self,
            model: Layer,
            optimizer: Optimizer,
            loss_function: LossFunction,
            train_loader: DataLoader,
            val_loader: DataLoader | None = None,
        ):
            self.model = model
            self.optimizer = optimizer
            self.loss_function = loss_function
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}


    def train_epoch(self) -> float:
        self.model.train = True

        total_loss = 0.0
        total_samples = 0

        for x_batch, t_batch in self.train_loader:
            # 順伝播
            out = self.model.forward_propagation(x_batch)
            loss = self.loss_function.forward_propagation(out, t_batch)

            batch_size = len(x_batch)

            total_loss += loss * batch_size
            total_samples += batch_size

            # 逆伝播
            dout = self.loss_function.backward_propagation(dout=1.0)
            self.model.backward_propagation(dout)

            # パラメータ更新
            self.optimizer.step()

        return total_loss / total_samples


    def evaluate(self) -> float:
        if self.val_loader is None:
            return 0.0

        self.model.train = False

        total_loss = 0.0
        total_samples = 0

        for x_batch, t_batch in self.val_loader:
            out = self.model.forward_propagation(x_batch)
            loss = self.loss_function.forward_propagation(out, t_batch)

            batch_size = len(x_batch)

            total_loss += loss * batch_size
            total_samples += batch_size

        return total_loss / total_samples


    def fit(self, epochs: int) -> dict[str, list[float]]:
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)

            if self.val_loader is not None:
                val_loss = self.evaluate()
                self.history["val_loss"].append(val_loss)
                print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f}")

        return self.history