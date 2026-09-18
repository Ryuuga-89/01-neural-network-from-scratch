from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    """
    損失関数の抽象基底クラス
    
    順伝播時にはモデルが出力した生の値と、正解データを受け取り、スカラーとして損失値を返す
    逆伝播時には上流からの勾配(デフォルトで1.0)を受け取り、モデル出力に対する勾配を返す。
    """
    loss: float
    output: np.ndarray
    t: np.ndarray
    
    def __init__(self):
        self.loss: float = 0.0

    @abstractmethod
    def forward_propagation(self, output: np.ndarray, t: np.ndarray) -> float: ...

    @abstractmethod
    def backward_propagation(self, dout: float = 1.0) -> np.ndarray: ...
    
    
class MeanSquaredErrorLoss(LossFunction):
    """
    二乗平均誤差
    回帰タスク用
    """
    
    def __init__(self):
        super().__init__()
    
    
    def forward_propagation(self, output: np.ndarray, t: np.ndarray) -> float:
        if output.shape != t.shape:
            raise ValueError("予測値と期待値の形状が異なります。")
        
        self.output = output
        self.t = t
        batch_size = self.output.shape[0] if self.output.ndim > 1 else 1

        # 二乗誤差の和をとってバッチサイズで割る
        self.loss = float(0.5 * np.sum((self.output - self.t) ** 2) / batch_size)
        return self.loss


    def backward_propagation(self, dout: float = 1.0) -> np.ndarray:
        batch_size = self.output.shape[0] if self.output.ndim > 1 else 1
        return dout * (self.output - self.t) / batch_size
    
    
class CrossEntropyLoss(LossFunction):
    """
    他クラス分類用
    入力 output: モデルの最終線形層出力(B, C)
    入力 t: 正解クラスインデックス(B,)またはone-hotラベル(B, C)
    """
    probs: np.ndarray

    def __init__(self):
        super().__init__()


    def forward_propagation(self, output: np.ndarray, t: np.ndarray) -> float:
        self.output = output
        self.t = t
        batch_size = output.shape[0] if output.ndim > 1 else 1

        # 数値安定な Softmax の計算 (行方向の最大値を減算)
        if output.ndim == 2:
            x_max = np.max(output, axis=1, keepdims=True)
            exp_x = np.exp(output - x_max)
            self.probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            x_max = np.max(output)
            exp_x = np.exp(output - x_max)
            self.probs = exp_x / np.sum(exp_x)

        # 損失の計算
        eps = 1e-15
        if self.t.size != self.probs.size:
            # ラベルがインデックス (B,) の場合
            log_p = np.log(self.probs[np.arange(batch_size), self.t] + eps)
            self.loss = float(-np.sum(log_p) / batch_size)
        else:
            # ラベルが one-hot (B, C) の場合
            self.loss = float(-np.sum(self.t * np.log(self.probs + eps)) / batch_size)

        return self.loss


    def backward_propagation(self, dout: float = 1.0) -> np.ndarray:
        batch_size = self.probs.shape[0] if self.probs.ndim > 1 else 1
        dx = self.probs.copy()

        if self.t.size != self.probs.size:
            dx[np.arange(batch_size), self.t] -= 1.0
        else:
            dx -= self.t

        return dout * (dx / batch_size)
    
    
class BCEWithLogitsLoss(LossFunction):
    """
    二値分類用(入力はSigmoid前のLogit)
    """
    sig_out: np.ndarray
    def __init__(self):
        super().__init__()


    def forward_propagation(self, output: np.ndarray, t: np.ndarray) -> float:
        assert output.shape == t.shape, f"Shape mismatch: {output.shape} vs {t.shape}"
        self.output = output
        self.t = t
        batch_size = output.shape[0] if output.ndim > 1 else 1

        # 安定なSigmoid: 1 / (1 + exp(-x))
        self.sig_out = np.where(
            output >= 0,
            1.0 / (1.0 + np.exp(-output)),
            np.exp(output) / (1.0 + np.exp(output))
        )

        # 安定なBCE計算: max(x, 0) - x * t + log(1 + exp(-|x|))
        loss_matrix = np.maximum(output, 0) - output * t + np.log(1.0 + np.exp(-np.abs(output)))
        self.loss = float(np.sum(loss_matrix) / batch_size)
        return self.loss


    def backward_propagation(self, dout: float = 1.0) -> np.ndarray:
        batch_size = self.sig_out.shape[0] if self.sig_out.ndim > 1 else 1
        return dout * (self.sig_out - self.t) / batch_size
