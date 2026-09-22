from typing import Literal

import numpy as np

from src.neural_networks.layers.abstract_layers import Layer, ParametricLayer


def padding_backward(dout_padded: np.ndarray, x_shape: tuple, padding_mode: str, padding_length: int) -> np.ndarray:
    """
    padding操作に対する逆伝播を行う。

    dout_padded: padding後のテンソルに対する勾配。shape = [B, C, H + 2P, W + 2P]

    x_shape: padding前の入力shape [B, C, H, W]

    Returns: padding前の入力に対する勾配 [B, C, H, W]
    """
    B, C, H, W = x_shape
    p = padding_length

    if p == 0:
        return dout_padded

    # Zero paddingの場合、padding部分は入力xに依存しないため
    # 中央部分だけを取り出せばよい
    if padding_mode == "Zeros":
        return dout_padded[
            :,
            :,
            p:p + H,
            p:p + W,
        ]

    PADDING_MODE_TO_NP_PAD: dict = {
        "Edge": "edge",
        "Reflect": "reflect",
        "Symmetric": "symmetric",
    }

    np_mode: Literal["edge", "reflect", "symmetric"] = PADDING_MODE_TO_NP_PAD[padding_mode]

    # 元画像の各pixelに一意なIDを割り当てる
    #
    # 例:
    # [[0, 1],
    #  [2, 3]]
    source_indices = np.arange(
        H * W,
        dtype=np.intp,
    ).reshape(H, W)

    # forward時と同じpaddingをindexに適用することで、padding後の各pixelが元画像のどこから来たのかを求める
    source_map = np.pad(
        source_indices,
        pad_width=((p, p), (p, p)),
        mode=np_mode,
    )

    source_flat = source_map.ravel()

    # [B, C, H * W]
    dx = np.zeros(
        (B, C, H * W),
        dtype=dout_padded.dtype,
    )

    # 同じ元pixelを参照しているpadding位置のgradientをnp.add.atによってすべて加算する
    for b in range(B):
        for c in range(C):
            np.add.at(dx[b, c], source_flat, dout_padded[b, c].ravel())

    return dx.reshape(B, C, H, W)

class Conv2D(ParametricLayer):
    """
    二次元畳み込み層
    
    shape = [C_out, C_in, Kh, Kw]
    
    argsの追加キー要件
    {
        ...
        
        "Conv2D": {
            stride: ストライド。int型
            padding_mode: パディング方法。"Zeros" / "Edge / "Reflect" / "Symmetric"
            padding_length: パディングする長さ。int型
        }
    }
    
    n_in = C_in * Kh * Kw
    n_out = C_out * Kh * Kw
    b_shape = (C_out,)
    """
    C_out: int
    C_in: int
    Kh: int
    Kw: int
    stride: int
    padding_mode: str
    padding_length: int
    x_col: np.ndarray
    x_shape: tuple
    
    @property
    def n_in(self) -> int:
        return self.C_in * self.Kh * self.Kw
    
    
    @property
    def n_out(self) -> int:
        return self.C_out * self.Kh * self.Kw
    
    
    @property
    def b_shape(self) -> tuple:
        return (self.C_out,)
    
    
    def __init__(self, shape: tuple, args: dict):
        self.C_out, self.C_in, self.Kh, self.Kw = shape
        super().__init__(shape, args)
        self.stride = self.args["Conv2D"]["stride"]
        self.padding_mode = self.args["Conv2D"]["padding_mode"]
        self.padding_length = self.args["Conv2D"]["padding_length"]
    
    
    def im2col(self, im: np.ndarray) -> np.ndarray:
        """
        4次元テンソルを、2次元行列に変換する。
        im: [B, C, H, W]をcol: [B * H_out * W_out, C_in * Kh * Kw]に変換する。
        """
        
        B, C, H, W = self.x_shape
        H_out = (H + 2 * self.padding_length - self.Kh) // self.stride + 1
        W_out = (W + 2 * self.padding_length - self.Kw) // self.stride + 1
        
        # インスタンス化時に指定されたpadding_modeをnp.pad関数の引数modeとして利用できるように変換する。
        PADDING_MODE_TO_NP_PAD: dict = {"Zeros": "constant", "Edge": "edge", "Reflect": "reflect", "Symmetric": "symmetric"}
        padding_mode_np_pad: Literal["constant", "edge", "reflect", "symmetric"] = PADDING_MODE_TO_NP_PAD[self.padding_mode]
        
        # imにpaddingを適用。"Zeros"が指定されている場合については、mode="constant"の際はデフォルトで0埋めなのを利用して実装している。
        im_padded = np.pad(im, pad_width=((0, 0), (0, 0), (self.padding_length, self.padding_length), (self.padding_length, self.padding_length)), mode=padding_mode_np_pad)
        col = np.zeros((B, C, self.Kh, self.Kw, H_out, W_out), dtype=im.dtype)
        
        # カーネルの各マス(i, j)について、計算対象となるim_paddedの要素を抽出する
        # この時点でのcolの定義は以下のとおり
        # col[b, c, i, j, oh, ow] = b番目の画像のcチャンネル目において、出力における(oh, ow)の位置の要素を計算するためのカーネルにおけるマス目(i, j)にあたっている入力配列の値
        for i in range(self.Kh):
            i_max = i + self.stride * H_out # 最初の位置はiであり、そこからH_out回のstride幅のジャンプをするとちょうどはみ出る
            for j in range(self.Kw):
                j_max = j + self.stride * W_out # 上と同じ。
                col[:, :, i, j, :, :] = im_padded[:, :, i:i_max:self.stride, j:j_max:self.stride] # はみ出る直前までstride幅でジャンプする
        
        col = col.transpose(0, 4, 5, 1, 2, 3) # col[B, H_out, W_out, C, Kh, Kw]
        col = col.reshape(B * H_out * W_out, -1) # col[B * H_out * W_out, C * Kh * Kw]
        
        return col
    
    
    def col2im(self, col: np.ndarray) -> np.ndarray:
        """
        2次元配列を4次元テンソルに変換する
        col: [B * H_out * W_out, C_in * Kh * Kw]をim: [B, C, H, W]に変換する
        """
        
        B, C, H, W = self.x_shape
        H_out = (H + 2 * self.padding_length - self.Kh) // self.stride + 1
        W_out = (W + 2 * self.padding_length - self.Kw) // self.stride + 1
        
        col = col.reshape(B, H_out, W_out, C, self.Kh, self.Kw)
        col = col.transpose(0, 3, 4, 5, 1, 2)
        im_padded: np.ndarray = np.zeros((B, C, H + 2 * self.padding_length, W + 2 * self.padding_length), dtype=col.dtype)
        
        for i in range(self.Kh):
            i_max = i + self.stride * H_out
            for j in range(self.Kw):
                j_max = j + self.stride * W_out
                im_padded[:, :, i:i_max:self.stride, j:j_max:self.stride] += col[:, :, i, j, :, :]
                
        return padding_backward(
            dout_padded=im_padded,
            x_shape=self.x_shape,
            padding_mode=self.padding_mode,
            padding_length=self.padding_length,
        )
        
    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        
        B, _, H, W = self.x_shape
        H_out = (H + 2 * self.padding_length - self.Kh) // self.stride + 1
        W_out = (W + 2 * self.padding_length - self.Kw) // self.stride + 1
        
        # xを[B, C_in, H, W] -> [B * H_out * W_out, C_in * Kh * Kw]
        self.x_col = self.im2col(x)
        
        # Wを[C_out, C_in, Kh, Kw] -> [C_in * Kh * Kw, C_out]
        W_col = self.W.reshape(self.shape[0], -1).T
        
        # Linearと同様の行列積計算 -> [B * H_out * W_out, C_out]
        out_col = self.x_col @ W_col + self.b
        
        # 元の形状に戻す
        # [B * H_out * W_out, C_out] -> [B, H_out, W_out, C_out] -> [B, C_out, H_out, W_out]
        out = out_col.reshape(B, H_out, W_out, -1).transpose(0, 3, 1, 2)
        
        return out
    
    
    def backward_propagation(self, dout: np.ndarray) -> np.ndarray:
        # 最終的に[B * H_out * W_out, C_out]
        dout_col = dout.transpose(0, 2, 3, 1).reshape(-1, self.b.shape[0])
        
        # Linearと同一の勾配計算
        self.dW = (self.x_col.T @ dout_col).T.reshape(self.W.shape)
        self.db = dout_col.sum(axis=0)
        
        dx_col = dout_col @ self.W.reshape(self.shape[0], -1)
        
        dx = self.col2im(dx_col)
        
        return dx
    
    
class MaxPooling2D(Layer):
    """
    二次元最大プーリング層
    
    shape = [Kh, Kw]
    
    argsの追加キー要件
    {
        ...
        
        "MaxPooling2D": {
            stride: ストライド。int型
            padding_mode: パディング方法。"Zeros" / "Edge" / "Reflect" / "Symmetric"
            padding_length: パディングする長さ。int型
        }
    }
    """
    shape: tuple
    args: dict
    Kh: int
    Kw: int
    stride: int
    padding_mode: str
    padding_length: int
    arg_max: np.ndarray
    x_shape: tuple
    
    
    def __init__(self, shape: tuple, args: dict):
        super().__init__()
        
        self.shape = shape
        self.args = args
        self.Kh, self.Kw = self.shape
        
        self.stride = self.args["MaxPooling2D"]["stride"]
        self.padding_mode = self.args["MaxPooling2D"]["padding_mode"]
        self.padding_length = self.args["MaxPooling2D"]["padding_length"]


    def im2col(self, im: np.ndarray) -> np.ndarray:
        B, C, H, W = self.x_shape
        H_out = (H + 2 * self.padding_length - self.Kh) // self.stride + 1
        W_out = (W + 2 * self.padding_length - self.Kw) // self.stride + 1
        
        PADDING_MODE_TO_NP_PAD: dict = {"Zeros": "constant", "Edge": "edge", "Reflect": "reflect", "Symmetric": "symmetric"}
        padding_mode_np_pad: Literal["constant", "edge", "reflect", "symmetric"] = PADDING_MODE_TO_NP_PAD[self.padding_mode]
        
        # Zerosパディングの場合、0埋めだと負の入力値に対して0が最大値として誤抽出されてしまうため -inf で埋める
        if padding_mode_np_pad == "constant":
            im_padded = np.pad(
                im, 
                pad_width=((0, 0), (0, 0), (self.padding_length, self.padding_length), (self.padding_length, self.padding_length)), 
                mode="constant", 
                constant_values=-np.inf
            )
        else:
            im_padded = np.pad(
                im, 
                pad_width=((0, 0), (0, 0), (self.padding_length, self.padding_length), (self.padding_length, self.padding_length)), 
                mode=padding_mode_np_pad
            )
            
        col = np.zeros((B, C, self.Kh, self.Kw, H_out, W_out), dtype=im.dtype)
        
        for i in range(self.Kh):
            i_max = i + self.stride * H_out
            for j in range(self.Kw):
                j_max = j + self.stride * W_out
                col[:, :, i, j, :, :] = im_padded[:, :, i:i_max:self.stride, j:j_max:self.stride]
        
        col = col.transpose(0, 4, 5, 1, 2, 3)
        col = col.reshape(B * H_out * W_out, -1)
        
        return col
    
    
    def col2im(self, col: np.ndarray) -> np.ndarray:
        """Conv2Dと完全に共通の処理"""
        B, C, H, W = self.x_shape
        H_out = (H + 2 * self.padding_length - self.Kh) // self.stride + 1
        W_out = (W + 2 * self.padding_length - self.Kw) // self.stride + 1
        
        col = col.reshape(B, H_out, W_out, C, self.Kh, self.Kw)
        col = col.transpose(0, 3, 4, 5, 1, 2)
        
        im_padded: np.ndarray = np.zeros((B, C, H + 2 * self.padding_length, W + 2 * self.padding_length), dtype=col.dtype)
        
        for i in range(self.Kh):
            i_max = i + self.stride * H_out
            for j in range(self.Kw):
                j_max = j + self.stride * W_out
                im_padded[:, :, i:i_max:self.stride, j:j_max:self.stride] += col[:, :, i, j, :, :]
                
        return padding_backward(
            dout_padded=im_padded,
            x_shape=self.x_shape,
            padding_mode=self.padding_mode,
            padding_length=self.padding_length,
        )


    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        B, C, H, W = self.x_shape
        H_out = (H + 2 * self.padding_length - self.Kh) // self.stride + 1
        W_out = (W + 2 * self.padding_length - self.Kw) // self.stride + 1
        
        # [B * H_out * W_out, C * Kh * Kw]に変換
        col = self.im2col(x)
        
        # パッチ内の各チャンネルごとに独立して最大値を取るためにリシェイプ
        # [B * H_out * W_out * C, Kh * Kw]
        col = col.reshape(-1, self.Kh * self.Kw)
        
        # 逆伝播のために最大値のインデックスを記憶
        self.arg_max = np.argmax(col, axis=1)
        
        # 横方向(Kh * Kw)に対して最大値を取得: [B * H_out * W_out * C]
        out_col = np.max(col, axis=1)
        
        # 元の特徴マップの形状に戻す: [B, C, H_out, W_out]
        out = out_col.reshape(B, H_out, W_out, C).transpose(0, 3, 1, 2)
        
        return out


    def backward_propagation(self, dout: np.ndarray) -> np.ndarray:
        _, C, _, _ = self.x_shape
        
        # 順伝播の `out_col` と同じ1次元形状に平坦化する
        # [B, C, H_out, W_out] -> [B, H_out, W_out, C] -> 1次元
        dout_flat = dout.transpose(0, 2, 3, 1).flatten()
        
        # [B * H_out * W_out * C, Kh * Kw]
        dx_col = np.zeros((dout_flat.size, self.Kh * self.Kw))
        
        # 最大値のインデックスの位置にだけ上流からの勾配を配置する
        # (np.arange で行を指定し、self.arg_max で列を指定して代入)
        dx_col[np.arange(self.arg_max.size), self.arg_max] = dout_flat
        
        # col2imの入力形状に戻す: [B * H_out * W_out, C * Kh * Kw]
        dx_col = dx_col.reshape(-1, C * self.Kh * self.Kw)
        
        # col2imでパッチを元の画像座標に集約する
        dx = self.col2im(dx_col)
        
        return dx