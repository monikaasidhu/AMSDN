import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction"""

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.out_channels = out_channels

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features):
        laterals = [
            lateral_conv(feat)
            for lateral_conv, feat in zip(self.lateral_convs, features)
        ]

        fpn_features = []
        prev_feat = laterals[-1]
        fpn_features.append(self.output_convs[-1](prev_feat))

        for i in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(
                prev_feat,
                size=laterals[i].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            prev_feat = laterals[i] + upsampled
            fpn_features.insert(0, self.output_convs[i](prev_feat))

        return fpn_features


class ConvNeXtFPN(nn.Module):
    """ConvNeXt-Tiny backbone with FPN for multi-scale features"""

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)  # ✅ FIXED (was 1,2,3,4)
        )

        self.feature_channels = self.backbone.feature_info.channels()

        self.fpn = FPN(
            in_channels_list=self.feature_channels,
            out_channels=256
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256 * 4, num_classes)

    def forward(self, x, return_features=False):
        backbone_features = self.backbone(x)
        fpn_features = self.fpn(backbone_features)

        if return_features:
            return fpn_features

        pooled_features = [
            self.global_pool(feat).flatten(1)
            for feat in fpn_features
        ]

        combined = torch.cat(pooled_features, dim=1)
        logits = self.classifier(combined)
        return logits

    def get_feature_dims(self):
        return [256] * 4


if __name__ == "__main__":
    model = ConvNeXtFPN(num_classes=10, pretrained=False)
    x = torch.randn(2, 3, 32, 32)

    logits = model(x)
    print("Logits shape:", logits.shape)

    features = model(x, return_features=True)
    print("Number of FPN levels:", len(features))
    for i, feat in enumerate(features):
        print(f"FPN level {i+1}: {feat.shape}")
