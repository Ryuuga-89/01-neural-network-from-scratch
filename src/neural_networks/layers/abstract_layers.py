from abc import ABC, abstractmethod
from math import sqrt
from typing import override

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
    def forward_propagation(self, x: np.ndarray) -> np.ndarray: ...
    
    @abstractmethod
    def backward_propagation(self, dout: np.ndarray) -> np.ndarray: ...
    
    # Optimizerに更新対象のパラメータを渡すためのメソッド
    # 基本は空のリストを返し、更新対象パラメータがある場合はオーバーライドする
    def get_params(self) -> list[tuple[object, str, str]]:
        return []
    
class ParametricLayer(Layer):
    """
    重み行列Wおよびバイアスベクトルbによって構成されるレイヤーを定義するための抽象クラス
    
    W, bの初期化を行う関数を共通でここで定義する。
    
    インスタンス化には以下の2つを要求する。
    shape: 重み行列の形状
    args: レイヤーの詳細を決定するのに必要な情報を格納した辞書
    
    args引数については、共通して以下の形式を要求する
    {
    "weight_init_method": {
        "method_name": 初期化方法("Standard" / "Xavier" / "He")
        "distribution": 初期化に用いる分布("normal" / "uniform")
        "seed": 乱数生成のためのシード値
        "sigma": [初期化方法がStandardで分布がnormalの場合のみ]標準偏差
        "r": [初期化方法がStandardで分布がuniformの場合のみ]一様分布の範囲
        }
        
    "bias_init_method": {
        "method_name": 初期化方法("Zeros")
        }
    }
    拡張性のためにこのようにしてある。
    他は、レイヤーごとに追加する。
    
    以下のプロパティを持つ
    shape: インスタンス化時に与えられたもの
    args: インスタンス化時に与えられたもの
    W: 重み行列
    b: バイアスベクトル
    dW: 誤差逆伝播時の重み行列による誤差
    db: 誤差逆伝播時のバイアスベクトルによる誤差
    n_in: 順伝播時に出力側の1ノードを計算するために寄与する入力要素の総数。抽象プロパティ
    n_out: 逆伝播時に入力側の1ノードを計算するために寄与する出力要素の総数。抽象プロパティ
    b_shape: バイアスベクトルの形状。抽象プロパティ
    
    n_in, n_out, n_shapeの3つのプロパティの実装が必要。
    """
    shape: tuple
    args: dict
    W: np.ndarray
    b: np.ndarray
    dW: np.ndarray
    db: np.ndarray

    # 順伝播時に出力側の1ノードを計算するために寄与する入力要素の総数
    @property
    @abstractmethod
    def n_in(self) -> int: ...
    
    # 逆伝播時に入力側の1ノードを計算するために寄与する出力要素の総数
    @property
    @abstractmethod
    def n_out(self) -> int: ...
    
    # バイアスベクトルの形状
    @property
    @abstractmethod
    def b_shape(self) -> tuple: ...
    
    def __init__(self, shape: tuple, args: dict):
        super().__init__()
        self.shape: tuple = shape
        self.args: dict = args
        self.W: np.ndarray = self.weight_init()
        self.b: np.ndarray = self.bias_init()
    
    
    def weight_init(self) -> np.ndarray:
        """
        Wの初期化方法として
        Standard-random-initialization / Xavier-initialization / He-initialization
        の3つを共通メソッドとして実装する。
        これらはself.args["weight_init_method"]["method_name"]によって指定されることが要求される。
        それぞれ Standard / Xavier / He が対応する。
        
        これらの違いは、正規分布あるいは一様分布を決定するためのパラメータのみなので、
        まず初期化方法ごとに分布を決めるパラメータを計算したのちに、それぞれの方法で初期化されたndarrayを返却するように実装する。
        """
        
        init_name: str = self.args["weight_init_method"]["method_name"]
        distribution: str = self.args["weight_init_method"]["distribution"]
        seed: int = self.args["weight_init_method"]["seed"]
        
        # 書いてから気づいたがもっといい実装があった。アルゴリズムはこっちの方がわかりやすいと思うのでこのままにする。
        if init_name == "Standard":
            if distribution == "normal" and self.args["weight_init_method"]["sigma"] >= 0:
                mean = 0
                sigma = self.args["weight_init_method"]["sigma"]
            elif distribution == "uniform" and self.args["weight_init_method"]["r"] >= 0:
                low = - self.args["weight_init_method"]["r"]
                high = self.args["weight_init_method"]["r"]
            else:
                raise ParameterInitializeMethodError("WのStandard初期化におけるargsの形式が不正")
            
        elif init_name == "Xavier":
            if distribution == "normal":
                mean = 0
                sigma = sqrt(2 / (self.n_in + self.n_out)) # 一回平方根にするの無駄だけど統一性のため
            elif distribution == "uniform":
                low = - sqrt(6 / (self.n_in + self.n_out))
                high = sqrt(6 / (self.n_in + self.n_out))
            else:
                raise ParameterInitializeMethodError("WのXavier初期化におけるargsの形式が不正")
        
        elif init_name == "He":
            if distribution == "normal":
                mean = 0
                sigma = sqrt(2 / self.n_in) # 一回平方根にするの無駄だけど統一性のため
            elif distribution == "uniform":
                low = - sqrt(6 / self.n_in)
                high = sqrt(6 / self.n_in)
            else:
                raise ParameterInitializeMethodError("WのHe初期化におけるargsの形式が不正")
        
        else:
            raise ParameterInitializeMethodError("Wの初期化方法がStandard / Xavier / Heのどれでもない")
        
        rng = np.random.default_rng(seed=seed)
        if distribution == "normal": return rng.normal(loc=mean, scale=sigma, size=self.shape)
        elif distribution == "uniform": return rng.uniform(low=low, high=high, size=self.shape)
        else: raise ParameterInitializeMethodError("通常ありえない箇所でのエラー")

    
    def bias_init(self) -> np.ndarray:
        init_name: str = self.args["bias_init_method"]["method_name"]
        
        if init_name == "Zeros":
            return np.zeros(self.b_shape)

        else:
            raise ParameterInitializeMethodError("bの初期化方法がZerosではない")
        
    
    @override
    def get_params(self) -> list[tuple[object, str, str]]:
        return [(self, "W", "dW"), (self, "b", "db")]