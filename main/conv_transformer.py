import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def weights_init_kaiming(m,  scale=1):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0.0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0.0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


class MHConvAttention(nn.Module):

    def __init__(self, num_heads=4, embedding_dim=64, out_dim=2048, window_size=5):
        super().__init__()
        self.nh = num_heads
        self.window_size = window_size
        self.pos_embed_dim = embedding_dim // self.nh
        self.rel_pos_embed = nn.Parameter(torch.zeros(self.pos_embed_dim, window_size, window_size))
        nn.init.normal_(self.rel_pos_embed, std=0.02)
        self.qkv_conv = nn.Conv2d(embedding_dim, 3 * embedding_dim, 1, bias=False)
        self.cpe = nn.Conv2d(embedding_dim, embedding_dim, 3, 1, 1, bias=False, groups=embedding_dim)
        self.qkv_conv.apply(weights_init_kaiming)
        self.cpe.apply(weights_init_kaiming)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.output_dim = out_dim
        self.out = nn.Sequential(
            nn.Conv2d(embedding_dim * 2, self.output_dim, 1, bias=False),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(inplace=True)
            )
        self.ChannelAttention = nn.Sigmoid()
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)

        self.out.apply(weights_init_kaiming)
        self.conv1d.apply(weights_init_kaiming)

    def forward(self, src):
        _, C, H, W = src.shape
        scaling_factor = (C // self.nh) ** -0.5
        feature_raw = src
        src = self.cpe(src) + src
        qkv = self.qkv_conv(src)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (nh hd) h w -> (b nh) hd h w", nh=self.nh)
        k = rearrange(k, "b (nh hd) h w -> (b nh) hd h w", nh=self.nh)
        v = rearrange(v, "b (nh hd) h w -> (b nh) hd h w", nh=self.nh)
        content_lambda = torch.einsum("bin, bon -> bio", k.flatten(-2).softmax(-1), v.flatten(-2))
        content_output = torch.einsum("bin, bio -> bon", q.flatten(-2) * scaling_factor, content_lambda)
        content_output = rearrange(content_output, "bnh hd (h w) -> bnh hd h w", h=H)
        position_lambda = F.conv2d(
            v,
            weight=rearrange(self.rel_pos_embed, "D Mx My -> D 1 Mx My"),
            padding=self.window_size // 2,
            groups=self.pos_embed_dim,
        )
        position_output = q * position_lambda
        result = content_output + position_output
        result1 = rearrange(result, "(b nh) hd h w -> b (nh hd) h w", nh=self.nh)
        X_1 = feature_raw
        X_1 = self.avg(X_1)
        X_1 = self.conv1d(X_1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        X_1 = self.ChannelAttention(X_1)
        X_1 = X_1.expand_as(feature_raw)
        result2 = feature_raw.mul(X_1)
        result = self.out(torch.cat([result1, result2], dim=1))
        return result
