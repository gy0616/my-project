import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarEnhancementModule(nn.Module):
    """
    Haar Enhancement Module (HEM)
    Enhances boundary precision using Haar wavelet transform in frequency domain.

    Architecture:
        DWT (Haar) → Channel Compression → Basic Block → Channel Recovery → IDWT
    """

    def __init__(self, channels, reduction_ratio=4):
        super().__init__()

        compressed_channels = max(channels // reduction_ratio, 4)

        # Compression for each sub-band
        self.compress_ll = self._make_compressor(channels, compressed_channels)
        self.compress_lh = self._make_compressor(channels, compressed_channels)
        self.compress_hl = self._make_compressor(channels, compressed_channels)
        self.compress_hh = self._make_compressor(channels, compressed_channels)

        # Non-linear processing
        self.process_block = nn.Sequential(
            nn.Conv2d(compressed_channels, compressed_channels, 3, padding=1,
                      groups=compressed_channels),
            nn.BatchNorm2d(compressed_channels),
            nn.GELU(),
            nn.Conv2d(compressed_channels, compressed_channels, 1),
            nn.BatchNorm2d(compressed_channels),
            nn.GELU()
        )

        # Recovery for each sub-band
        self.expand_ll = nn.Conv2d(compressed_channels, channels, 1)
        self.expand_lh = nn.Conv2d(compressed_channels, channels, 1)
        self.expand_hl = nn.Conv2d(compressed_channels, channels, 1)
        self.expand_hh = nn.Conv2d(compressed_channels, channels, 1)

        # Register Haar kernels
        self.register_buffer('haar_kernel', self._get_haar_kernel(channels))
        self.register_buffer('inv_haar_kernel', self._get_inverse_haar_kernel(channels))

        # Learnable residual weight
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Output normalization
        self.out_norm = nn.BatchNorm2d(channels)
        self.out_act = nn.ReLU(inplace=True)

    def _make_compressor(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def _get_haar_kernel(self, channels):
        LL = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        LH = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
        HL = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32)
        HH = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        base_kernels = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)
        return base_kernels.repeat(channels, 1, 1, 1)

    def _get_inverse_haar_kernel(self, channels):
        inv_LL = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        inv_LH = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
        inv_HL = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32)
        inv_HH = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        base_kernels = torch.stack([inv_LL, inv_LH, inv_HL, inv_HH], dim=0).unsqueeze(1)
        return base_kernels.repeat(channels, 1, 1, 1)

    def dwt(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2), mode='reflect')

        x_dwt = F.conv2d(x, self.haar_kernel, stride=2, groups=C)
        B_, C4, Hh, Ww = x_dwt.shape
        x_dwt = x_dwt.reshape(B_, C, 4, Hh, Ww)

        return (x_dwt[:, :, 0, :, :], x_dwt[:, :, 1, :, :],
                x_dwt[:, :, 2, :, :], x_dwt[:, :, 3, :, :])

    def idwt(self, ll, lh, hl, hh):
        x_stacked = torch.cat([ll, lh, hl, hh], dim=1)
        return F.conv_transpose2d(x_stacked, self.inv_haar_kernel, stride=2,
                                  groups=ll.shape[1])

    def forward(self, x):
        if x.shape[2] < 4 or x.shape[3] < 4:
            return x

        residual = x
        ll, lh, hl, hh = self.dwt(x)

        # Process each sub-band
        ll_enhanced = self.process_block(self.compress_ll(ll))
        lh_enhanced = self.process_block(self.compress_lh(lh))
        hl_enhanced = self.process_block(self.compress_hl(hl))
        hh_enhanced = self.process_block(self.compress_hh(hh))

        # Restore channels
        ll_restored = self.expand_ll(ll_enhanced)
        lh_restored = self.expand_lh(lh_enhanced)
        hl_restored = self.expand_hl(hl_enhanced)
        hh_restored = self.expand_hh(hh_enhanced)

        reconstructed = self.idwt(ll_restored, lh_restored, hl_restored, hh_restored)
        out = residual + self.alpha * reconstructed
        out = self.out_norm(out)
        return self.out_act(out)