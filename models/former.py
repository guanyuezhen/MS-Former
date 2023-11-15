import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class P2M_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_mode='p2t'):
        super().__init__()
        self.num_heads = num_heads
        self.pool_mode = pool_mode
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        #
        if pool_mode == 'p2t':
            self.avg_pool_s1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.avg_pool_s2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.avg_pool_s3 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.avg_pool_s4 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.max_pool = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        else:
            pass
        self.norm_p = nn.LayerNorm(dim)
        #
        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, m, H, W, patch_size):
        B, L, C = m.shape
        q = self.q(m).reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        if self.pool_mode == 'p2t':
            avg_pool_s1 = F.adaptive_avg_pool2d(x_, (round(H * 4 / 12), round(H * 4 / 12)))
            avg_pool_s2 = F.adaptive_avg_pool2d(x_, (round(H * 4 / 16), round(H * 4 / 16)))
            avg_pool_s3 = F.adaptive_avg_pool2d(x_, (round(H * 4 / 20), round(H * 4 / 20)))
            avg_pool_s4 = F.adaptive_avg_pool2d(x_, (round(H * 4 / 24), round(H * 4 / 24)))
            max_pool = F.adaptive_max_pool2d(x_, (H * 4 // patch_size, W * 4 // patch_size))
            avg_pool_s1 = avg_pool_s1 + self.avg_pool_s1(avg_pool_s1)
            avg_pool_s2 = avg_pool_s2 + self.avg_pool_s2(avg_pool_s2)
            avg_pool_s3 = avg_pool_s3 + self.avg_pool_s3(avg_pool_s3)
            avg_pool_s4 = avg_pool_s4 + self.avg_pool_s4(avg_pool_s4)
            max_pool = max_pool + self.max_pool(max_pool)
            p = torch.cat([avg_pool_s1.reshape(B, C, -1).permute(0, 2, 1),
                           avg_pool_s2.reshape(B, C, -1).permute(0, 2, 1),
                           avg_pool_s3.reshape(B, C, -1).permute(0, 2, 1),
                           avg_pool_s4.reshape(B, C, -1).permute(0, 2, 1),
                           max_pool.reshape(B, C, -1).permute(0, 2, 1)
                           ], dim=1
                          )
            p = self.norm_p(p)
        else:
            pass

        kv = self.kv(
            torch.cat([p, m], dim=1)
        ).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        m = (attn @ v).transpose(1, 2).reshape(B, L, C)
        m = self.proj(m)
        m = self.proj_drop(m)

        return m, max_pool


class M2P_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        #
        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, m):
        B, N, C = x.shape
        B, L, C = m.shape
        #
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(m).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.p2m_attn = P2M_Attention(dim, num_heads=num_heads)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.m2t_attn = M2P_Attention(dim, num_heads=num_heads)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path_x = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_m = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm5 = norm_layer(dim)
        self.norm6 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_x = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.mlp_m = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, m, H, W, patch_size):
        m_, max_pool = self.p2m_attn(self.norm1(x), self.norm2(m), H, W, patch_size)
        m = m + self.drop_path_m(m_)
        x_ = self.m2t_attn(self.norm3(x), self.norm4(m))
        x = x + self.drop_path_x(x_)
        #
        m = m + self.drop_path_m(self.mlp_m(self.norm5(m)))
        x = x + self.drop_path_x(self.mlp_x(self.norm6(x)))
        return x, m, max_pool
