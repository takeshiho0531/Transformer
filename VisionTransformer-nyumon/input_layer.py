import torch
import torch.nn as nn


class VitInputLayer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        emb_dim: int = 384,
        num_patch_row: int = 2,
        image_size: int = 32,
    ):
        """
        Args:
            in_channels (int, optional): 入力画像のチャネル数. Defaults to 3.
            emb_dim (int, optional): 埋め込み後のベクトルの長さ. Defaults to 384.
            num_patch_row (int, optional): 高さ方向のパッチ数. Defaults to 2. 例は2*2であるため、2をデフォルト値にした。
            image_size (int, optional): 入力画像の一辺の大きさ. Defaults to 32.
        """
        super(VitInputLayer, self).__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.num_patch_row = num_patch_row
        self.image_size = image_size
