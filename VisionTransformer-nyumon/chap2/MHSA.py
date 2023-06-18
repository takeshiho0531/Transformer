# Multi-Head Self-Attentionの実装
import torch
import torch.nn as nn
import torch.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim: int=384, head: int=3, dropout: float=0.):
        """_summary_

        Args:
            emb_dim (int, optional): 埋め込み後のベクトルの長さ Defaults to 384.
            head (int, optional): ヘッドの数. Defaults to 3.
            dropout (float, optional):ドロップアウト率. Defaults to 0..
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.head=head
        self.emb_dim=emb_dim
        self.head_dim=emb_dim//head
        self.sqrt_dh=self.head_dim**0.5  # D_hの二乗根。qk^Tを割るための係数

        # 入力をq,k,vに埋め込むための線形層
        self.w_q=nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k=nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v=nn.Linear(emb_dim, emb_dim, bias=False)

        #　式7にはないが、実装ではdropout層も用いる
        self.attn_drop=nn.Dropout(dropout)

        # MHSAの結果を出力に埋め込むための線形層
        # 式10にはないが実装ではdropout層にも用いる
        self.w_o=nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            z (torch.Tensor): MHSAへの入力。形状は(B, N, D)
                            B: バッチサイズ, N: トークン数, D: ベクトルの長さ

        Returns:
            torch.Tensor: MHSAの出力。形状は(B, N, D)
                            B: バッチサイズ, N: トークン数, D: 埋め込みベクトルの長さ
        """
        batch_size, num_patch=z.size()

        # 埋め込み
        ## (B,N,D)->(B,N,D)
        q=self.w_q(z)
        k=self.w_k(z)
        v=self.w_v(z)

        # q,k,vをヘッドに分ける
        ## まずベクトルをヘッドの個数(h)に分ける
        ## (B,N,D)->(B,N,h,D//h)
        q=q.view(batch_size, num_patch, self.head, self.head_dim)
        k=k.view(batch_size, num_patch, self.head, self.head_dim)
        v=v.view(batch_size, num_patch, self.head, self.head_dim)

        ## self-attentionができるように
        ## (バッチサイズ、ヘッド、トークン数、パッチのベクトル)の形に変更する
        ## (B,N,h,D//h)->(B,h,N,D//h)
        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)

        # 内積
        k_T=k.transpose(2,3)
        dots=(q@k_T)/self.sqrt_dh
        attn=F.softmax(dots, dim=-1)
        attn=self.attn_drop(attn)

        out=attn@v
        out=out.transpose(1,2)
        out=out.reshape(batch_size, num_patch, self.emb_dim)

        out=self.w_o(out)
        return out
