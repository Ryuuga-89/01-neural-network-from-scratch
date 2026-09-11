from typing import Any

import numpy as np

from src.neural_networks.layers.abstract_layers import Layer, ParametricLayer


class Linear(ParametricLayer):
    """
    線形層
    shape = [in, out]
    n_in = in
    n_out = out
    b_shape = (out,)
    """
    x: np.ndarray # 誤差逆伝播用に入力を保存する。[..., n_in]
    
    @property
    def n_in(self) -> int:
        return self.shape[0]
    
    @property
    def n_out(self) -> int:
        return self.shape[1]
    
    @property
    def b_shape(self) -> tuple:
        return (self.shape[1],)
    
    def __init__(self, shape: tuple, args: dict):
        super().__init__(shape, args)
    
    
    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        """
        Linearの順伝播Ax+bを計算する
        xは[..., in]の形式
        """
        
        self.x = x
        x_shape = x.shape # [B, C, n_in]など。次元数は2次元以上が要求。
        
        x_2d: np.ndarray = self.x.reshape(-1, self.n_in) # [X, n_in]の形に
        ans: np.ndarray = x_2d @ self.W + self.b # [X, n_in] @ [n_in, n_out] + [n_out]
        ans = ans.reshape(*x_shape[:-1], -1)
        return ans
    
    
    def backward_propagation(self, dout: np.ndarray) -> np.ndarray:
        """
        dxを返すとともに、dWとdbを計算する
        doutは[..., out]の形式
        """
        
        x: np.ndarray = self.x # 入力を取り出す。
        
        dout_shape = dout.shape # [B, C, n_out]など。次元数は2次元以上が要求。
        
        dout_2d: np.ndarray = dout.reshape(-1, self.n_out) # [X, n_out]の形
        x_2d: np.ndarray = x.reshape(-1, self.n_in) # [X, n_in]の形
        
        self.dW = x_2d.T @ dout_2d
        self.db = dout_2d.sum(axis=0)
        
        dx_2d = dout_2d @ self.W.T
        
        dx = dx_2d.reshape(*dout_shape[:-1], -1)
        
        return dx


class ReLU(Layer):
    """
    ReLU関数
    
    インスタンス化にあたっての引数は不要
    """
    mask: np.ndarray # 0より大きいかどうかの真偽地を格納した行列
    
    
    def __init__(self):
        super().__init__()
    
    
    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        """
        0より大きいかどうかを確認し、結果をmaskに格納する
        maskとxのアダマール積が出力になる
        """
        self.mask = x > 0
        return x * self.mask
    
    
    def backward_propagation(self, dout: np.ndarray) -> np.ndarray:
        """
        順伝播時に0より大きかった場合は勾配をそのまま流し、
        0以下であった場合は勾配を0にする
        """
        return dout * self.mask


class Sigmoid(Layer):
    """
    Sigmoid関数
    
    インスタンス化にあたっての引数は不要
    """
    out: np.ndarray # 出力。誤差逆伝播時の計算に必要。
    
    def __init__(self):
        super().__init__()
        
    
    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        self.out = 1.0 / (1.0 + np.exp(x))
        return self.out
    
    
    def backward_propagation(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.out * (1.0 - self.out)
    
    
