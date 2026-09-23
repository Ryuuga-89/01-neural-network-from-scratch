from typing import override

import numpy as np

from src.neural_networks.layers.abstract_layers import Layer


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
    
    
class BatchNormalization(Layer):
    """
    バッチ正規化層
    パラメータを持っているとも言えるが、初期化方法が異なるため、ParametricLayerではなくLayerを継承している。
    
    2Dでも4Dでも対応できるようにしている。これによりLinearの後でもConv2Dの後でも使える
    
    args = {
        "BatchNormalization": {
            "channels": 正規化するチャンネル数
            "momentum": 指数移動平均(EMA)用の係数
            "eps": 正規化時に0除算を回避するための微小な数
        }
    }
    """
    C: int
    mu_infer: np.ndarray # 推論用の平均
    var_infer: np.ndarray # 推論用の分散
    gamma: np.ndarray # スケール用のパラメータ。学習される。
    beta: np.ndarray # シフト用のパラメータ。学習される。
    dgamma: np.ndarray # gemmaの勾配
    dbeta: np.ndarray # betaの勾配
    
    x_c: np.ndarray # 平均移動した後のx
    std: np.ndarray # xの標準偏差
    invs: np.ndarray # 標準偏差の逆数
    xn: np.ndarray # 正規化されたx
    
    # 2Dと4Dの処理共通化のため
    param_shape: tuple # ブロードキャスト用に変換するためのshape
    axis: tuple # 平均をとる方向。2DならB方向、4DならB, H, W方向
    x_shape: tuple # xの形状保存
    
    
    def __init__(self, args: dict):
        super().__init__()
        self.C = args["BatchNormalization"]["channels"]
        self.momentum = args["BatchNormalization"]["momentum"]
        self.eps = args["BatchNormalization"]["eps"]
        
        self.gamma = np.ones(self.C)
        self.beta = np.zeros(self.C)
        
        self.mu_infer = np.zeros(self.C)
        self.var_infer = np.ones(self.C)


    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        
        if x.ndim == 2:
            self.axis = (0,)
            param_shape = (1, -1)
        elif x.ndim == 4:
            self.axis = (0, 2, 3)
            param_shape = (1, -1, 1, 1)
        else:
            raise ValueError("BatchNorm expects 2D or 4D tensor.")

        gamma_bc = self.gamma.reshape(param_shape)
        beta_bc = self.beta.reshape(param_shape)
        
        if not self.train:
            mu = self.mu_infer.reshape(param_shape)
            var = self.var_infer.reshape(param_shape)
            x_c = x - mu
            std = np.sqrt(var + self.eps)
            xn = x_c / std
            out = gamma_bc * xn + beta_bc
            return out
        
        # 1. 平均
        mu = np.mean(x, axis=self.axis, keepdims=True)
        
        # 2. センタリング
        self.x_c = x - mu
        
        # 3. 分散
        var = np.mean(self.x_c ** 2, axis=self.axis, keepdims=True)
        
        # 4. 標準偏差
        self.std = np.sqrt(var + self.eps)
        
        # 5. 標準偏差の逆数
        self.invs = 1.0 / self.std
        
        # 6. 正規化
        self.xn = self.x_c * self.invs
        
        # 7. シフト＆スケール
        out = gamma_bc * self.xn + beta_bc
        
        # 移動平均の更新
        self.mu_infer = self.momentum * self.mu_infer + (1 - self.momentum) * mu.flatten()
        self.var_infer = self.momentum * self.var_infer + (1 - self.momentum) * var.flatten()
        
        self.param_shape = param_shape
        
        return out


    def backward_propagation(self, dout: np.ndarray) -> np.ndarray:
        # gamma の形状をブロードキャスト用に変形
        gamma_bc = self.gamma.reshape(self.param_shape)
        N = np.prod([self.x_shape[i] for i in self.axis])
        
        # 7. y = gamma * xn + beta
        self.dgamma = np.sum(dout * self.xn, axis=self.axis)
        self.dbeta = np.sum(dout, axis=self.axis)
        dxn = dout * gamma_bc
        
        # 6. xn = x_c * invs 
        dinvs = np.sum(dxn * self.x_c, axis=self.axis, keepdims=True)
        dx_c1 = dxn * self.invs
        
        # 5. invs = 1 / std = std^(-1)
        # -1 * std^(-2) を掛ける
        dstd = dinvs * (-1.0 / (self.std ** 2))
        
        # 4. std = sqrt(var + eps) = (var + eps)^(1/2)
        # 0.5 * (var + eps)^(-1/2) を掛ける
        dvar = dstd * 0.5 / self.std
        
        # 3. var = mean(x_c^2)
        # x^2の微分2*xを掛けて、meanなのでNで割る
        dx_c2 = dvar * (2.0 / N) * self.x_c
        
        # x_cの勾配を合流させる
        dx_c = dx_c1 + dx_c2
        
        # 2. x_c = x - mu
        # xへの直接の経路と、muへの経路に分岐する
        dmu = np.sum(dx_c * (-1.0), axis=self.axis, keepdims=True)
        dx1 = dx_c  # ( dx_c * 1.0 )
        
        # 1. mu = mean(x)
        dx2 = dmu * (1.0 / N)
        
        # xの勾配を合流させる
        dx = dx1 + dx2
        
        return dx
    
    
    @override
    def get_params(self) -> list[tuple[object, str, str]]:
        return [(self, "gamma", "dgamma"), (self, "beta", "dbeta")]
    
class Sequential(Layer):
    """
    層を順番にまとめるためのもの
    全てのモデルはSequentialを用いて定義されるものとする
    """
    layers: list[Layer]
    _train: bool # trainのsetter/getterを変えるのでその代わりに別の変数を利用している
    
    
    def __init__(self, layers: list[Layer]):
        self.layers = layers
        self._train = True
        super().__init__()
        
        
    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        """
        全ての層の順伝播メソッドを通した上で、出力する
        """
        for layer in self.layers:
            x = layer.forward_propagation(x)
            
        return x
    
    
    def backward_propagation(self, dout: np.ndarray) -> np.ndarray:
        """
        全ての層の逆伝播メソッドを通した上で、出力する
        出力の理由はないが、メソッド的な統一性のために出力することにする
        """
        for layer in reversed(self.layers):
            dout = layer.backward_propagation(dout)
            
        return dout
    
    
    @property
    def train(self) -> bool:
        return self._train
    
    
    @train.setter
    def train(self, mode: bool) -> None:
        self._train = mode
        for layer in self.layers:
            layer.train = mode
            
            
    @override
    def get_params(self) -> list[tuple[object, str, str]]:
        params = []
        
        for layer in self.layers:
            params.extend(layer.get_params())
            
        return params