import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """Spatial attention to focus on important regions"""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            attention map [B, 1, H, W] and attended features [B, C, H, W]
        """
        # Aggregate across channels
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # [B, 1, H, W]
        
        # Concatenate and convolve
        concat = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, H, W]
        attention = self.sigmoid(self.conv(concat))  # [B, 1, H, W]
        
        # Apply attention
        out = x * attention
        
        return attention, out


class ChannelAttention(nn.Module):
    """Channel attention to emphasize important feature channels"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            attention weights [B, C, 1, 1] and attended features [B, C, H, W]
        """
        B, C, _, _ = x.size()
        
        # Global pooling
        avg = self.avg_pool(x).view(B, C)
        max_val = self.max_pool(x).view(B, C)
        
        # Shared FC layers
        avg_out = self.fc(avg)
        max_out = self.fc(max_val)
        
        # Combine and activate
        attention = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        
        # Apply attention
        out = x * attention
        
        return attention, out


class MultiScalePyramidAttention(nn.Module):
    """Multi-scale pyramid attention for scale-adaptive processing"""
    
    def __init__(self, channels):
        super().__init__()
        
        # Pyramid branches with different dilation rates
        self.branch1 = nn.Conv2d(channels, channels//4, 1, bias=False)
        self.branch2 = nn.Conv2d(channels, channels//4, 3, padding=1, bias=False)
        self.branch3 = nn.Conv2d(channels, channels//4, 3, padding=2, dilation=2, bias=False)
        self.branch4 = nn.Conv2d(channels, channels//4, 3, padding=4, dilation=4, bias=False)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            Multi-scale fused features [B, C, H, W]
        """
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # Concatenate all branches
        concat = torch.cat([b1, b2, b3, b4], dim=1)
        
        # Fuse and add residual
        out = self.fusion(concat) + x
        
        return out


class AdaptiveAttentionModule(nn.Module):
    """
    Unified adaptive attention combining:
    - Spatial attention (where to focus)
    - Channel attention (what to emphasize)
    - Multi-scale pyramid attention (scale awareness)
    """
    
    def __init__(self, channels):
        super().__init__()
        
        self.spatial_attn = SpatialAttention()
        self.channel_attn = ChannelAttention(channels)
        self.pyramid_attn = MultiScalePyramidAttention(channels)
        
        # Adaptive fusion weights (learnable)
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            attended features [B, C, H, W] and attention maps dict
        """
        # Apply all attention mechanisms
        spatial_map, spatial_out = self.spatial_attn(x)
        channel_map, channel_out = self.channel_attn(x)
        pyramid_out = self.pyramid_attn(x)
        
        # Normalize fusion weights
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # Adaptive fusion
        out = (weights[0] * spatial_out + 
               weights[1] * channel_out + 
               weights[2] * pyramid_out)
        
        # Attention maps for visualization/analysis
        attention_maps = {
            'spatial': spatial_map,
            'channel': channel_map,
            'fusion_weights': weights
        }
        
        return out, attention_maps


class MultiScaleAdaptiveAttention(nn.Module):
    """Apply adaptive attention across all FPN levels"""
    
    def __init__(self, fpn_channels=256, num_levels=4):
        super().__init__()
        
        # Attention module for each FPN level
        self.attention_modules = nn.ModuleList([
            AdaptiveAttentionModule(fpn_channels)
            for _ in range(num_levels)
        ])
        
    def forward(self, fpn_features):
        """
        Args:
            fpn_features: List of FPN feature maps [P2, P3, P4, P5]
        Returns:
            attended_features: List of attended features
            all_attention_maps: List of attention maps for each level
        """
        attended_features = []
        all_attention_maps = []
        
        for feat, attn_module in zip(fpn_features, self.attention_modules):
            attended_feat, attn_maps = attn_module(feat)
            attended_features.append(attended_feat)
            all_attention_maps.append(attn_maps)
        
        return attended_features, all_attention_maps


# Test function
if __name__ == "__main__":
    # Test single attention module
    attn = AdaptiveAttentionModule(channels=256)
    x = torch.randn(2, 256, 8, 8)
    out, maps = attn(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"Spatial map: {maps['spatial'].shape}")
    print(f"Fusion weights: {maps['fusion_weights']}")
    
    # Test multi-scale attention
    ms_attn = MultiScaleAdaptiveAttention(fpn_channels=256, num_levels=4)
    fpn_feats = [torch.randn(2, 256, 8, 8) for _ in range(4)]
    attended, all_maps = ms_attn(fpn_feats)
    print(f"\nMulti-scale attended features: {len(attended)}")
    for i, feat in enumerate(attended):
        print(f"Level {i}: {feat.shape}")