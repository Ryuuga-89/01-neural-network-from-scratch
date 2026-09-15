from typing import Any

import numpy as np

from src.neural_networks.layers.abstract_layers import Layer, ParametricLayer


class Dropout(Layer):
    """
    ドロップアウトを処理する層
    
    args = {
        Dropout: {
            dropout_rate: ドロップアウトさせる確率p
            seed: 乱数シード
        }
    }
    """
    dropout_rate: float
    seed: int
    rng: np.random.Generator
    mask: np.ndarray
    
    def __init__(self, args: dict):
        super().__init__()
        
        self.dropout_rate = args["Dropout"]["dropout_rate"]
        self.seed = args["Dropout"]["seed"]
        
        self.rng = np.random.default_rng(self.seed)
    
    
    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        """
        xと同型のマスク配列を作成し、指定された確率で0のマスと1のマスを生成
        """
        # 推論モードならそのまま流す
        if not self.train:
            return x
        
        
        keep_p = 1.0 - self.dropout_rate
        
        # 一様分布からサンプリングし、大小関係を使ってマスキングとスケーリングを行う。
        self.mask = (self.rng.random(x.shape) < keep_p) / keep_p
        
        out = x * self.mask
        
        return out
    
    
    def backward_propagation(self, dout: np.ndarray) -> np.ndarray:
        """
        保存してあるマスクを用いて、0にしたところは勾配を0にして流し、そうでないところはスケールを直して流す
        """
        
        # 通常あり得ないが念の為
        if not self.train or self.mask is None:
            return dout
            
        # 順伝播で用いたスケーリング済みのマスクを、上流からの勾配にもそのまま掛ける
        dx = dout * self.mask
        return dx