from abc import ABC, abstractmethod

import numpy as np

from src.neural_networks.layers.abstract_layers import Layer


class Optimizer(ABC):
    """
    オプティマイザーを定義するための抽象クラス
    
    """
    params: list[tuple[object, str, str]]
    
    def __init__(self, layer: Layer, lr: float = 0.01):
        self.params = layer.get_params()
        self.lr = lr
        
        
    def step(self) -> None:
        """
        更新対象パラメータをupdateメソッド内で全て更新する。
        勾配の初期化を行うzero_gradを呼び出す。
        """
        self.update()
        
        self.zero_grad()
        
    
    @abstractmethod
    def update(self) -> None:
        """更新の責務を負う"""
        ...


    def zero_grad(self) -> None:
        """各パラメータの勾配をゼロで初期化する"""
        for layer, param_name, grad_name in self.params:
            if hasattr(layer, grad_name):
                grad = getattr(layer, grad_name, None)
                if isinstance(grad, np.ndarray):
                    grad.fill(0) # ゼロ埋め
                else:
                    raise ValueError("パラメータ更新時なのにも関わらず定義されていない勾配が存在します。")
                
                
class SGD(Optimizer):
    def __init__(self, layer: Layer, lr: float = 0.01):
        super().__init__(layer, lr)


    def update(self) -> None:
        for layer, param_name, grad_name in self.params:
            grad = getattr(layer, grad_name, None)
            if grad is None:
                continue

            param = getattr(layer, param_name)
            # インプレース減算で更新（参照を維持）
            param -= self.lr * grad
            
            
class Momentum(Optimizer):
    momentum: float
    v: list[np.ndarray]
    
    
    def __init__(self, layer: Layer, lr: float = 0.01, momentum: float = 0.9):
        super().__init__(layer, lr)
        self.momentum = momentum
        self.v = [
                    np.zeros_like(getattr(target_layer, param_name))
                    for target_layer, param_name, _ in self.params
                ]


    def update(self) -> None:
        for idx, (layer, param_name, grad_name) in enumerate(self.params):
            grad = getattr(layer, grad_name, None)
            if grad is None:
                continue

            param = getattr(layer, param_name)

            # v = momentum * v + grad
            self.v[idx] = self.momentum * self.v[idx] + grad
            # param = param - lr * v
            param -= self.lr * self.v[idx]
            
            
class AdaGrad(Optimizer):
    h: list[np.ndarray]
    eps: float


    def __init__(self, layer: Layer, lr: float = 0.01, eps: float = 1e-8):
        super().__init__(layer, lr)
        self.eps = eps
        # 過去の勾配二乗和を蓄積する配列を各パラメータと同じ shape で初期化
        self.h = [
            np.zeros_like(getattr(target_layer, param_name))
            for target_layer, param_name, _ in self.params
        ]


    def update(self) -> None:
        for idx, (layer, param_name, grad_name) in enumerate(self.params):
            grad = getattr(layer, grad_name, None)
            if grad is None:
                continue

            param = getattr(layer, param_name)

            # 勾配の二乗和を蓄積
            self.h[idx] += grad * grad
            # パラメータの更新
            param -= (self.lr / (np.sqrt(self.h[idx]) + self.eps)) * grad
            
            
class Adam(Optimizer):
    beta1: float
    beta2: float
    eps: float
    iter: int
    m: list[np.ndarray]
    v: list[np.ndarray]

    def __init__(self, layer: Layer, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(layer, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.iter = 0

        # 1次モーメント(m)と 2次モーメント(v)をゼロ配列で初期化
        self.m = [
            np.zeros_like(getattr(target_layer, param_name))
            for target_layer, param_name, _ in self.params
        ]
        self.v = [
            np.zeros_like(getattr(target_layer, param_name))
            for target_layer, param_name, _ in self.params
        ]

    def update(self) -> None:
        self.iter += 1

        for idx, (layer, param_name, grad_name) in enumerate(self.params):
            grad = getattr(layer, grad_name, None)
            if grad is None:
                continue

            param = getattr(layer, param_name)

            self.m[idx] = self.beta1 * self.m[idx] + (1.0 - self.beta1) * grad
            self.v[idx] = self.beta2 * self.v[idx] + (1.0 - self.beta2) * grad ** 2

            m_hat = self.m[idx] / (1.0 - self.beta1 ** self.iter)
            v_hat = self.v[idx] / (1.0 - self.beta2 ** self.iter)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
