import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """Spatial attention for feature refinement"""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(y))


class EfficientAttentionGate(nn.Module):
    """Efficient attention gate for feature fusion"""

    def __init__(self, F_g, F_l, F_int, num_groups=8):
        super().__init__()
        g_groups = min(num_groups, max(1, F_g))
        x_groups = min(num_groups, max(1, F_l))

        self.grouped_conv_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, groups=g_groups),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )
        self.grouped_conv_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, groups=x_groups),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        g1 = self.grouped_conv_g(g)
        x1 = self.grouped_conv_x(x)
        psi = self.psi(self.relu(x1 + g1))
        return x * psi + x


class EfficientChannelAttention(nn.Module):
    """Efficient channel attention (ECA)"""

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        v = v.squeeze(-1).transpose(-1, -2)
        v = self.conv(v)
        v = v.transpose(-1, -2).unsqueeze(-1)
        return self.sigmoid(v)


class SemanticGuidedDecouplingAggregation(nn.Module):


    def __init__(self, in_channels, num_classes, mid_channels=None, num_groups=8):
        super().__init__()
        mid = mid_channels if mid_channels is not None else max(in_channels // 2, 1)

        # Efficient Attention Gate
        self.eag = EfficientAttentionGate(in_channels, in_channels, mid, num_groups)

        # Inter-Class Decoupling
        self.class_attn_generator = nn.Sequential(
            nn.Conv2d(in_channels * 2, num_classes, 1),
            nn.Softmax(dim=1)
        )
        self.background_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, 1, 1),
            nn.Sigmoid()
        )
        self.class_aggregation = nn.Sequential(
            nn.Conv2d(in_channels * num_classes, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Efficient Channel Attention
        self.eca = EfficientChannelAttention(in_channels * 2)

        # Final projection
        self.project = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, global_feat, local_feat):
        """
        Args:
            global_feat: Features from Swin Transformer branch (global context)
            local_feat: Features from ConvNeXt branch (local details)
        Returns:
            fused: Fused features
            class_attn: Class attention maps for visualization
        """
        # Step 1: Adaptive fusion via EAG
        gated_local = self.eag(global_feat, local_feat)

        # Align spatial dimensions
        if global_feat.shape[2:] != gated_local.shape[2:]:
            global_feat = F.interpolate(global_feat, size=gated_local.shape[2:],
                                        mode='bilinear', align_corners=False)

        # Step 2: Inter-Class Decoupling
        concat = torch.cat([gated_local, global_feat], dim=1)
        class_attn = self.class_attn_generator(concat)
        bg_gate = self.background_gate(concat)

        # Extract class-specific features
        class_features = []
        for c in range(class_attn.shape[1]):
            c_attn = class_attn[:, c:c + 1, :, :] * (1 - bg_gate)
            c_feat = gated_local * c_attn
            class_features.append(c_feat)

        # Aggregate decoupled features
        decoupled = torch.cat(class_features, dim=1)
        decoupled_out = self.class_aggregation(decoupled)
        enhanced_local = gated_local + self.alpha * decoupled_out

        # Step 3: Channel attention
        final_concat = torch.cat([enhanced_local, global_feat], dim=1)
        ca_weight = self.eca(final_concat)
        final_concat = final_concat * ca_weight

        # Step 4: Final projection
        return self.project(final_concat), class_attn