import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

plt = None


# ---------------------------
# 条带卷积 (strip conv)
# ---------------------------
class StripConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, k), padding=(0, k // 2))
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(k, 1), padding=(k // 2, 0))
        self.fuse = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return self.fuse(torch.cat([x1, x2], dim=1))


# ---------------------------
# 改进版通道注意力 (并行分支)
# ---------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        mid_channels = max(8, in_channels // reduction)

        self.gap_fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        )
        self.gmp_fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        )
        self.conv1d = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.freq_fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape

        gap = F.adaptive_avg_pool2d(x, 1)
        gap_out = self.gap_fc(gap)

        gmp = F.adaptive_max_pool2d(x, 1)
        gmp_out = self.gmp_fc(gmp)

        freq = x.view(b, c, -1)
        freq = self.conv1d(freq).mean(-1).view(b, c, 1, 1)
        freq_out = self.freq_fc(freq)

        out = gap_out + gmp_out + freq_out
        return self.sigmoid(out)  # [B,C,1,1]


# ---------------------------
# 空间注意力
# ---------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        x_cat = torch.cat([max_pool, avg_pool], dim=1)
        return self.sigmoid(self.conv(x_cat))  # [B,1,H,W]


# ---------------------------
# HFDSF / HLFIG 模块（带可视化保存）
# ---------------------------
class HFDSF(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, num_heads=4):
        super().__init__()
        deep_channels = in_channels[0]
        shallow_channels = in_channels[1]
        mid_channels = shallow_channels // reduction

        self.reduce1 = nn.Conv2d(shallow_channels, mid_channels, kernel_size=1)
        self.reduce2 = nn.Conv2d(deep_channels, mid_channels, kernel_size=1)

        self.strip5 = StripConv(mid_channels, mid_channels, k=5)
        self.strip7 = StripConv(mid_channels, mid_channels, k=7)

        self.q_proj = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.mhsa = nn.MultiheadAttention(embed_dim=mid_channels, num_heads=num_heads, batch_first=True)

        self.channel_att = ChannelAttention(mid_channels)
        self.spatial_att = SpatialAttention()

        self.alpha_conv = nn.Conv2d(mid_channels * 2, 2, kernel_size=1)
        self.out_proj = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

        # ✅ 建议仍然初始化，但即使没初始化，下面也能 lazy init 补上
        self._vis_enabled = False
        self._vis_save_dir = None
        self._vis_prefix = "img"
        self._vis_sample_idx = 0
        self._vis_mode = "absmean"
        self._vis_cmap = "jet"
        self._vis_once = True
        self._vis_already_saved = False
        self._vis_skip = 0

    # -----------------------
    # ✅ 惰性初始化（关键）
    # -----------------------
    def _lazy_init_vis(self):
        if not hasattr(self, "_vis_enabled"):
            self._vis_enabled = False
        if not hasattr(self, "_vis_save_dir"):
            self._vis_save_dir = None
        if not hasattr(self, "_vis_prefix"):
            self._vis_prefix = "img"
        if not hasattr(self, "_vis_sample_idx"):
            self._vis_sample_idx = 0
        if not hasattr(self, "_vis_mode"):
            self._vis_mode = "absmean"
        if not hasattr(self, "_vis_cmap"):
            self._vis_cmap = "jet"
        if not hasattr(self, "_vis_once"):
            self._vis_once = True
        if not hasattr(self, "_vis_already_saved"):
            self._vis_already_saved = False

    def set_vis(self,
                enabled: bool,
                save_dir: str,
                prefix: str = "img",
                sample_idx: int = 0,
                mode: str = "absmean",
                cmap: str = "jet",
                once: bool = True,
                skip=0):
        self._vis_skip = int(skip)  # 或者直接作为参数

        self._lazy_init_vis()
        self._vis_enabled = enabled
        self._vis_save_dir = save_dir
        self._vis_prefix = prefix
        self._vis_sample_idx = int(sample_idx)
        self._vis_mode = mode
        self._vis_cmap = cmap
        self._vis_once = once
        if not enabled:
            self._vis_already_saved = False

    @staticmethod
    def _to_heatmap(x_bchw: torch.Tensor, sample_idx: int, mode: str, crop: int = 2) -> np.ndarray:
        """
        x_bchw: [B,C,H,W]
        crop: 裁掉边界像素数（默认 2），用于消除边界高亮/伪影
        """
        x = x_bchw[sample_idx].detach().float().cpu()  # [C,H,W]

        # ---- ✅ 裁边：去掉四周 crop 个像素 ----
        if crop is not None and crop > 0 and x.dim() == 3:
            H, W = x.shape[-2], x.shape[-1]
            # 防止图太小导致空
            if H > 2 * crop + 1 and W > 2 * crop + 1:
                x = x[:, crop:-crop, crop:-crop]

        # ---- 聚合通道得到 2D heatmap ----
        if x.dim() == 2:
            hm = x
        else:
            if mode == "mean":
                hm = x.mean(dim=0)
            elif mode == "max":
                hm = x.max(dim=0)[0]
            else:  # absmean
                hm = x.abs().mean(dim=0)

        hm = hm.numpy()
        hm = hm - hm.min()
        hm = hm / (hm.max() + 1e-12)
        return hm

    @staticmethod
    def _save_hm(hm: np.ndarray, path: str, cmap: str, dpi: int = 200):
        global plt
        if plt is None:
            import matplotlib.pyplot as plt  # noqa: PLC0415
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.figure(figsize=(4, 4), dpi=dpi)
        plt.axis("off")
        plt.imshow(hm, cmap=cmap)
        plt.tight_layout(pad=0)
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close()

    def _save_feature_maps(self, shallow: torch.Tensor, deep: torch.Tensor, fused: torch.Tensor):
        # ✅ 关键：先 lazy init，避免缺属性崩溃
        if self._vis_skip > 0:
            self._vis_skip -= 1
            return

        self._lazy_init_vis()

        if (not self._vis_enabled) or (self._vis_save_dir is None):
            return
        if self._vis_once and self._vis_already_saved:
            return

        maps = {"shallow": shallow, "deep": deep, "fused": fused}
        for k, v in maps.items():
            hm = self._to_heatmap(v, self._vis_sample_idx, self._vis_mode)
            out_path = os.path.join(self._vis_save_dir, f"{self._vis_prefix}_{k}.png")
            self._save_hm(hm, out_path, self._vis_cmap)

        self._vis_already_saved = True

    def forward(self, x_in):
        deep, shallow = x_in[0], x_in[1]
        if deep.shape[-2:] != shallow.shape[-2:]:
            deep = F.interpolate(deep, size=shallow.shape[-2:], mode="bilinear", align_corners=False)
        B, _, H, W = shallow.shape

        x1 = self.reduce1(shallow)
        x2 = self.reduce2(deep)

        K = self.strip5(x1)
        V = self.strip7(x1)
        Q = self.q_proj(x2)

        Q_ = Q.flatten(2).permute(0, 2, 1)
        K_ = K.flatten(2).permute(0, 2, 1)
        V_ = V.flatten(2).permute(0, 2, 1)

        attn_out, _ = self.mhsa(Q_, K_, V_)
        attn_out = attn_out.permute(0, 2, 1).reshape(B, -1, H, W)

        ch_weight = self.channel_att(x2)
        shallow_att = x1 * ch_weight

        sp_weight = self.spatial_att(attn_out)
        deep_att = x2 * sp_weight

        fusion_in = torch.cat([shallow_att, deep_att], dim=1).to(self.alpha_conv.weight.dtype)
        alpha = F.softmax(self.alpha_conv(fusion_in), dim=1)
        out_mid = alpha[:, 0:1] * shallow_att + alpha[:, 1:2] * deep_att

        out = self.out_proj(out_mid)

        # 保存三种特征（浅/深/融合）
        self._save_feature_maps(shallow=x1, deep=x2, fused=out_mid)
        return out
