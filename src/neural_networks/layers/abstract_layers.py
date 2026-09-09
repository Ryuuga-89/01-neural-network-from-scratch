from abc import ABC, abstractmethod
from math import sqrt

import numpy as np

from src.errors.errors import ParameterInitializeMethodError


class Layer(ABC):
    """
    レイヤーを定義するための抽象クラス
    順伝播、逆伝播を抽象メソッドとし、実装を強制させる。
    """
    train: bool # 学習モード(true)か推論モード(false)かを管理。使わない場合もある。
    
    def __init__(self):
        self.train: bool = True # インスタンス時には学習モード
    
    @abstractmethod
    def forward_propagation(self): ...
    
    @abstractmethod
    def backward_propagation(self): ...
    
    
    
class ParametricLayer(Layer):
    """
    パラメータを要求するレイヤーを定義するための抽象クラス
    ここでパラメータとは、誤差逆伝播法により更新されるものをいう。
    
    さらにパラメータ初期化メソッドをここで定義しておく
    パラメータ初期化メソッドにおいてはn_inとn_outが必要になるが、これはレイヤーの種類により計算方法が異なるので、計算方法のみ抽象化する。
    プロパティは以下のように定義する。
    """
    shape: tuple # 形状を決定するのに必要な情報。インスタンス化時に与える。
    initial_method: str # 初期化方法 Standard / Xavier / Heのいずれか。インスタンス化時に与える。
    args: dict # 初期化分布あるいはレイヤーの詳細を決定するのに必要な情報。インスタンス化時に与える。
               # argsは、initial_method = standardだった場合にのみ、一様分布あるいは正規分布を決定するために必要な引数(それぞれr, sigma)を持つ
    params: np.ndarray # パラメータ

    # 順伝播時に出力側の1ノードを計算するために寄与する入力要素の総数
    @property
    @abstractmethod
    def n_in(self) -> int: ...
    
    # 逆伝播時に入力側の1ノードを計算するために寄与する出力要素の総数
    @property
    @abstractmethod
    def n_out(self) -> int: ...
    
    def __init__(self, shape: tuple, initial_method: str, args: dict):
        super().__init__()
        self.shape: tuple = shape
        self.initial_method: str = initial_method
        self.args: dict = args
        self.params: np.ndarray = self.param_init()
    
    def param_init(self) -> np.ndarray:
        """
        パラメータ初期化方法として
        Standard-random-initialization / Xavier-initialization / He-initialization
        の3つを共通メソッドとして実装する。
        これらはmethod引数によって指定される。
        それぞれ Standard / Xavier / He が対応する。
        
        これらの違いは、正規分布あるいは一様分布を決定するためのパラメータのみなので、
        まず初期化方法ごとに分布を決めるパラメータを計算したのちに、初期化されたndarrayを返却するように実装する。
        """
        
        # 書いてから気づいたがもっといい実装があった。アルゴリズムはこっちの方がわかりやすいと思うのでこのままにする。
        if self.initial_method == "Standard":
            if self.args["distribution"] == "normal" and self.args["sigma"] >= 0:
                mean = 0
                sigma = self.args["sigma"]
            elif self.args["distribution"] == "uniform" and self.args["r"] >= 0:
                low = - self.args["r"]
                high = self.args["r"]
            else:
                raise ParameterInitializeMethodError("Standard初期化におけるargsの形式が不正")
            
        elif self.initial_method == "Xavier":
            if self.args["distribution"] == "normal":
                mean = 0
                sigma = sqrt(2 / (self.n_in + self.n_out)) # 一回平方根にするの無駄だけど統一性のため
            elif self.args["distribution"] == "uniform":
                low = - sqrt(6 / (self.n_in + self.n_out))
                high = sqrt(6 / (self.n_in + self.n_out))
            else:
                raise ParameterInitializeMethodError("Xavier初期化におけるargsの形式が不正")
        
        elif self.initial_method == "He":
            if self.args["distribution"] == "normal":
                mean = 0
                sigma = sqrt(2 / self.n_in) # 一回平方根にするの無駄だけど統一性のため
            elif self.args["distribution"] == "uniform":
                low = - sqrt(6 / self.n_in)
                high = sqrt(6 / self.n_in)
            else:
                raise ParameterInitializeMethodError("He初期化におけるargsの形式が不正")
        
        else:
            raise ParameterInitializeMethodError("初期化方法がStandard / Xavier / Heのどれでもない")
        
        rng = np.random.default_rng(seed=42)
        if self.args["distribution"] == "normal": return rng.normal(loc=mean, scale=sigma**2, size=self.shape)
        elif self.args["distribution"] == "uniform": return rng.uniform(low=low, high=high, size=self.shape)
        else: raise ParameterInitializeMethodError("通常ありえない箇所でのエラー")
