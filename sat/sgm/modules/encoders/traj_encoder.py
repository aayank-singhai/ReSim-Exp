# modified from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_1d.py

import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from sgm.modules.encoders.modules import AbstractEmbModel
import numpy as np

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)

        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return self.mlp_head(cls_tokens)

class TrajEncoder(AbstractEmbModel):
    def __init__(self, *, seq_len, dim, out_dim, depth, mlp_dim, heads=8, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., 
                 pos_emb="learnable",
                 avoid_first_ln=False,
                 use_all_tokens=False,
                 zero_init=False):
        super().__init__()
        # assert (seq_len % patch_size) == 0

        # num_patches = seq_len // patch_size
        # patch_dim = channels * patch_size

        if avoid_first_ln:
            self.to_patch_embedding = nn.Sequential(
                nn.Linear(channels, dim),
                nn.LayerNorm(dim),
            )
        else:
            self.to_patch_embedding = nn.Sequential(
                # Rearrange('b c (n p) -> b n (p c)', p = patch_size),
                nn.LayerNorm(channels),   # * Do we need this?
                nn.Linear(channels, dim),
                nn.LayerNorm(dim),
            )
        
        assert pos_emb in ["learnable", "sine"]

        # learn_pos = nn.Parameter(torch.randn(1, seq_len + 1, dim))
        # grid_t = np.arange(1 + seq_len, dtype=np.float32)  # +1 for cls token
        # sine_pos = get_1d_sincos_pos_embed_from_grid(dim, grid_t)
        # sine_pos = torch.tensor(sine_pos).unsqueeze(0)
        # import pdb; pdb.set_trace()

        if pos_emb == "learnable":
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, dim))
        else:
            # TODO: Sine pos emb 1d
            grid_t = np.arange(1 + seq_len, dtype=np.float32)  # +1 for cls token
            sine_pos = get_1d_sincos_pos_embed_from_grid(dim, grid_t)
            self.pos_embedding = torch.tensor(sine_pos).unsqueeze(0)

        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )
        if zero_init:
            nn.init.zeros_(self.mlp_head[1].weight)
            nn.init.zeros_(self.mlp_head[1].bias)
        self.use_all_tokens = use_all_tokens

    def forward(self, series):
        series = series.to(torch.float16)
        x = self.to_patch_embedding(series)  # [2, 8, 3]  b, t, c
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)

        x, ps = pack([cls_tokens, x], 'b * d')

        pos = self.pos_embedding[:, :(n + 1)]
        pos = pos.to(x)
        x += pos
        x = self.dropout(x)

        x = self.transformer(x)  # [2, 9, 1024]

        if self.use_all_tokens:
            out = self.mlp_head(x)
        else:
            cls_tokens, _ = unpack(x, ps, 'b * d')
            out = self.mlp_head(cls_tokens)
            out = out.unsqueeze(1)  # [2, 1, 1024]

        return out


if __name__ == '__main__':

    # v = ViT(
    #     seq_len = 256,
    #     patch_size = 16,
    #     num_classes = 1000,
    #     dim = 1024,
    #     depth = 6,
    #     heads = 8,
    #     mlp_dim = 2048,
    #     dropout = 0.1,
    #     emb_dropout = 0.1
    # )
    
    v = TrajEncoder(
        seq_len = 8,
        dim = 1024,
        out_dim = 1024,
        pos_emb = 'sine',
        depth = 3,
        heads = 8,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1,
        channels=3,
    )

    time_series = torch.randn(4, 8, 3)
    logits = v(time_series) # (4, 1024)
    # import pdb; pdb.set_trace()