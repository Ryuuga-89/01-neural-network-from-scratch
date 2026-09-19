import math
from collections.abc import Iterator

import numpy as np


class DataLoader:
    """
    データセット(X, t)からミニバッチを生成するイテレータクラス。
    インスタンス化時には特徴量、正解データともに全サンプルを並べたnp配列を要求する。
    また各種設定も要求する
    
    __len__と__iter__を実装し、イテレータープロトコルに準拠させる
    """
    X: np.ndarray # 特徴量。形式は(サンプル数, ...)
    t: np.ndarray # 正解データ。形式は(サンプル数, ...)
    batch_size: int # バッチサイズ
    shuffle: bool # エポックごとにデータを並び替えるかどうか
    drop_last: bool # データ数がバッチサイズで割り切れない場合、あまりをどうするか
    seed: int | None # 乱数シード

    def __init__(
        self,
        X: np.ndarray,
        t: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        if len(X) != len(t):
            raise ValueError("特徴量と正解データのサンプル数が一致しません。")
        if batch_size <= 0:
            raise ValueError("バッチサイズは正の整数である必要があります。")

        self.X = X
        self.t = t
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

        self.num_samples = len(X)


    def __len__(self) -> int:
        """
        1エポックあたりのバッチ数を返す
        """
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)


    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        イテレーション開始時にインデックスを準備し、バッチを順次yieldする
        """
        # インデックス配列の生成
        if self.shuffle:
            indices = self.rng.permutation(self.num_samples)
        else:
            indices = np.arange(self.num_samples)

        # drop_last の場合は端数を切り捨て
        max_len = (
            (self.num_samples // self.batch_size) * self.batch_size
            if self.drop_last
            else self.num_samples
        )

        for start_idx in range(0, max_len, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]

            # ファンシーインデックス参照でスライスを生成
            yield self.X[batch_indices], self.t[batch_indices]