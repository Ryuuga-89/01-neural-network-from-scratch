import numpy as np

from typing import Literal

from src.neural_networks.layers.abstract_layers import Layer, ParametricLayer


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
        super().__init__(shape, args)
        self.C_out, self.C_in, self.Kh, self.Kw = self.shape
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
                
        if self.padding_length == 0:
            im = im_padded
        else:
            im = im_padded[:, :, self.padding_length:-self.padding_length, self.padding_length:-self.padding_length]
        
        return im
        
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
    
    
