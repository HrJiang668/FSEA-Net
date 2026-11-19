# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Frequency evidence allocation modules for FSEA-Net."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CFE(nn.Module):
    """Adaptive frequency extraction used by FSEA-Net.

    The module receives three feature levels, aligns them to the shallow level,
    fuses their complex spectra with scale-aware weights, allocates the fused
    spectrum into high/mid/low bands, and returns three residual-enhanced
    features with the same shapes as the inputs.
    """

    def __init__(self, in_chs=(128, 256, 512), mid_ch=256):
        super().__init__()
        self.in_chs = tuple(in_chs)
        self.mid_ch = int(mid_ch)

        self.proj = nn.ModuleList(nn.Conv2d(c, self.mid_ch, 1, bias=False) for c in self.in_chs)
        self.back = nn.ModuleList(nn.Conv2d(self.mid_ch, c, 1, bias=False) for c in self.in_chs)

        self.spectral_norm = nn.InstanceNorm2d(self.mid_ch * 3, affine=False)
        self.gamma_conv = nn.Conv2d(self.mid_ch * 3, 3, 1, bias=True)
        self.spatial_mask = nn.Sequential(
            nn.Conv2d(self.mid_ch * 3, self.mid_ch, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.mid_ch, 1, 3, padding=1, bias=False),
            nn.Sigmoid(),
        )

        # Learnable radial-mask parameters. Softplus/sigmoid keep them in valid ranges.
        self.sigma_high = nn.Parameter(torch.tensor(0.25))
        self.sigma_mid = nn.Parameter(torch.tensor(0.18))
        self.sigma_low = nn.Parameter(torch.tensor(0.45))
        self.r_mid = nn.Parameter(torch.tensor(0.42))

        # Small identity-preserving initialization for safe training from YOLOv8 weights.
        self.band_gain = nn.Parameter(torch.ones(3) * 0.10)
        self.residual_gate = nn.Parameter(torch.ones(3) * -2.5)
        self.eps = 1e-6

    def _bands(self, h, w, device):
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=torch.float32).view(h, 1)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=torch.float32).view(1, w)
        r = torch.sqrt(xx.square() + yy.square()).clamp_(0.0, 1.0).unsqueeze(0).unsqueeze(0)

        sigma_h = F.softplus(self.sigma_high) + self.eps
        sigma_m = F.softplus(self.sigma_mid) + self.eps
        sigma_l = F.softplus(self.sigma_low) + self.eps
        r0 = self.r_mid.sigmoid()

        high = 1.0 - torch.exp(-(r.square()) / (2.0 * sigma_h.square()))
        mid = torch.exp(-((r - r0).square()) / (2.0 * sigma_m.square()))
        low = torch.exp(-(r.square()) / (2.0 * sigma_l.square()))
        return high, mid, low

    @staticmethod
    def _magnitude_descriptor(freq):
        return torch.log1p(torch.abs(freq))

    def forward(self, xs):
        x1, x2, x3 = xs
        out_dtype = x1.dtype
        h, w = x1.shape[-2:]

        p1 = self.proj[0](x1).float()
        p2 = F.interpolate(self.proj[1](x2).float(), size=(h, w), mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.proj[2](x3).float(), size=(h, w), mode="bilinear", align_corners=False)

        f1 = torch.fft.fftshift(torch.fft.fft2(p1, norm="ortho"), dim=(-2, -1))
        f2 = torch.fft.fftshift(torch.fft.fft2(p2, norm="ortho"), dim=(-2, -1))
        f3 = torch.fft.fftshift(torch.fft.fft2(p3, norm="ortho"), dim=(-2, -1))

        desc = torch.cat(
            [self._magnitude_descriptor(f1), self._magnitude_descriptor(f2), self._magnitude_descriptor(f3)], dim=1
        )
        desc = self.spectral_norm(desc)
        desc_conv = desc.to(dtype=self.gamma_conv.weight.dtype)

        gamma = self.gamma_conv(desc_conv).softmax(dim=1).float()
        fused = (
            gamma[:, 0:1].expand_as(f1) * f1
            + gamma[:, 1:2].expand_as(f2) * f2
            + gamma[:, 2:3].expand_as(f3) * f3
        )

        spatial = self.spatial_mask(desc_conv).float().expand(-1, self.mid_ch, -1, -1)
        bands = self._bands(h, w, x1.device)
        gains = F.softplus(self.band_gain)
        gates = self.residual_gate.sigmoid()

        outputs = []
        refs = (x1, x2, x3)
        sizes = (x1.shape[-2:], x2.shape[-2:], x3.shape[-2:])
        for i, band in enumerate(bands):
            band = band.expand_as(fused)
            freq = fused * (gains[i].view(1, 1, 1, 1) * band) * spatial
            feat = torch.fft.ifft2(torch.fft.ifftshift(freq, dim=(-2, -1)), norm="ortho").real
            feat = self.back[i](feat.to(dtype=self.back[i].weight.dtype))
            if feat.shape[-2:] != sizes[i]:
                feat = F.interpolate(feat, size=sizes[i], mode="bilinear", align_corners=False)
            outputs.append(refs[i] + gates[i].view(1, 1, 1, 1) * feat.to(out_dtype))

        return tuple(outputs)
