import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .hem import HaarEnhancementModule
from .sgda import SemanticGuidedDecouplingAggregation
from .udt import UncertaintyAwareDualTraining

class ChannelAdapter(nn.Module):
    """Adapter for aligning feature dimensions"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DUHANet(nn.Module):

    def __init__(self, num_classes=6, feature_dim=128,
                 convnext_name='convnext_tiny',
                 swin_name='swin_tiny_patch4_window7_224',
                 img_size=256,
                 haar_stages=[0, 1, 2, 3],
                 haar_reduction_ratio=4):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # ==================== Dual Backbones ====================
        self.local_backbone = timm.create_model(
            convnext_name, pretrained=True, features_only=True,
            out_indices=[0, 1, 2, 3]
        )
        self.global_backbone = timm.create_model(
            swin_name, pretrained=True, features_only=True,
            out_indices=[0, 1, 2, 3], img_size=img_size
        )

        # ==================== Feature Adapters ====================
        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size)
            local_feats = self.local_backbone(dummy)
            global_feats = self.global_backbone(dummy)

        self.local_adapters = nn.ModuleList([
            ChannelAdapter(f.shape[1], feature_dim) for f in local_feats
        ])
        self.global_adapters = nn.ModuleList([
            ChannelAdapter(f.shape[1], feature_dim) for f in global_feats
        ])

        # ==================== Haar Enhancement Module (HEM) ====================
        self.hem = nn.ModuleList([
            HaarEnhancementModule(feature_dim, haar_reduction_ratio)
            if i in haar_stages else nn.Identity()
            for i in range(4)
        ])

        # ==================== Semantic-Guided Decoupling Aggregation (S-GDA) ====================
        self.sgda = nn.ModuleList([
            SemanticGuidedDecouplingAggregation(feature_dim, num_classes)
            for _ in range(4)
        ])

        # ==================== FPN Decoder ====================
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])

        # ==================== Segmentation Head ====================
        self.seg_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(feature_dim // 2, num_classes, 1)
        )

        # Auxiliary heads for UDT
        self.local_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, num_classes, 1)
        )

        self.global_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, num_classes, 1)
        )

    def forward(self, x):
        """Standard forward pass"""
        H, W = x.shape[2:]

        local_feats = self.local_backbone(x)
        global_feats = self.global_backbone(x)

        fused_feats = []
        last_local = last_global = None

        for i in range(4):
            # Local branch with HEM
            local = self.local_adapters[i](local_feats[i])
            local = self.hem[i](local)

            # Global branch
            global_feat = self.global_adapters[i](global_feats[i])
            if global_feat.dim() == 3:
                B, N, C = global_feat.shape
                Hw = int(N ** 0.5)
                global_feat = global_feat.permute(0, 2, 1).reshape(B, C, Hw, Hw)

            if global_feat.shape[2:] != local.shape[2:]:
                global_feat = F.interpolate(global_feat, size=local.shape[2:],
                                            mode='bilinear', align_corners=False)

            if i == 3:
                last_local = local
                last_global = global_feat

            fused, _ = self.sgda[i](global_feat, local)
            fused = self.fpn_convs[i](fused)
            fused_feats.append(fused)

        # FPN aggregation
        out = fused_feats[-1]
        for i in range(2, -1, -1):
            out = F.interpolate(out, size=fused_feats[i].shape[2:],
                                mode='bilinear', align_corners=False)
            out = out + fused_feats[i]

        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        # Final output
        seg_out = self.seg_head(out)

        # Auxiliary outputs for UDT
        local_out = self.local_head(last_local)
        global_out = self.global_head(last_global)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        global_out = F.interpolate(global_out, size=(H, W), mode='bilinear', align_corners=False)

        return {
            'segmentation': seg_out,
            'local_logits': local_out,
            'global_logits': global_out,
        }

    def forward_with_features(self, x):
        """Forward pass returning features for uncertainty computation"""
        output = self.forward(x)
        return output['segmentation'], {
            'local_logits': output['local_logits'],
            'global_logits': output['global_logits'],
        }


# ==================== Utility Functions ====================
def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(num_classes=6, img_size=256, **kwargs):
    """Factory function to create DUHA-Net model"""
    model = DUHANet(
        num_classes=num_classes,
        img_size=img_size,
        **kwargs
    )
    return model