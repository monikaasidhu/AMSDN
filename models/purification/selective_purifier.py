import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyDetector(nn.Module):
    """Detects adversarial perturbations in feature space"""
    
    def __init__(self, channels):
        super().__init__()
        
        # Feature statistics analyzer
        self.analyzer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: Feature map [B, C, H, W]
        Returns:
            anomaly_score: [B, 1] - higher means more likely adversarial
        """
        score = self.analyzer(x)
        return score


class FeaturePurifier(nn.Module):
    """Purifies features using denoising autoencoder structure"""
    
    def __init__(self, channels):
        super().__init__()
        
        # Encoder path
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Decoder path
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        """
        Args:
            x: Noisy/adversarial features [B, C, H, W]
        Returns:
            purified features [B, C, H, W]
        """
        # Encode and decode
        encoded = self.encoder(x)
        purified = self.decoder(encoded)
        
        # Residual connection
        purified = purified + x
        
        return purified


class SelectivePurificationModule(nn.Module):
    """
    Selective purification in feature space:
    - Detects anomalous regions
    - Selectively purifies based on anomaly score
    - Preserves clean regions
    """
    
    def __init__(self, channels, threshold=0.5):
        super().__init__()
        
        self.threshold = threshold
        
        # Detection branch
        self.detector = AnomalyDetector(channels)
        
        # Purification branch
        self.purifier = FeaturePurifier(channels)
        
        # Gating mechanism for selective application
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, return_scores=False):
        """
        Args:
            x: Feature map [B, C, H, W]
            return_scores: If True, return detection scores
        Returns:
            purified features [B, C, H, W]
            (optional) anomaly_scores [B, 1]
        """
        # Detect anomalies
        anomaly_scores = self.detector(x)  # [B, 1]
        
        # Purify features
        purified = self.purifier(x)
        
        # Generate spatial gating map
        gate_map = self.gate(x)  # [B, C, H, W]
        
        # Selective fusion based on anomaly score
        # High score -> more purification, Low score -> keep original
        alpha = anomaly_scores.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        
        # Blend original and purified based on detection + spatial gating
        output = alpha * (gate_map * purified + (1 - gate_map) * x) + (1 - alpha) * x
        
        if return_scores:
            return output, anomaly_scores
        return output
    
    def is_adversarial(self, x):
        """
        Binary decision: is input adversarial?
        Args:
            x: Feature map [B, C, H, W]
        Returns:
            Boolean tensor [B]
        """
        scores = self.detector(x).squeeze()
        return scores > self.threshold


class MultiScaleSelectivePurifier(nn.Module):
    """Selective purification across all FPN levels"""
    
    def __init__(self, fpn_channels=256, num_levels=4, threshold=0.5):
        super().__init__()
        
        # Purification module for each level
        self.purifiers = nn.ModuleList([
            SelectivePurificationModule(fpn_channels, threshold)
            for _ in range(num_levels)
        ])
        
    def forward(self, fpn_features, return_scores=False):
        """
        Args:
            fpn_features: List of FPN feature maps
            return_scores: If True, return anomaly scores
        Returns:
            purified_features: List of purified features
            (optional) all_scores: List of anomaly scores per level
        """
        purified_features = []
        all_scores = []
        
        for feat, purifier in zip(fpn_features, self.purifiers):
            if return_scores:
                purified, scores = purifier(feat, return_scores=True)
                all_scores.append(scores)
            else:
                purified = purifier(feat)
            purified_features.append(purified)
        
        if return_scores:
            return purified_features, all_scores
        return purified_features
    
    def detect_adversarial(self, fpn_features):
        """
        Aggregate detection across all levels
        Args:
            fpn_features: List of FPN feature maps
        Returns:
            is_adversarial: [B] boolean tensor
        """
        all_scores = []
        for feat, purifier in zip(fpn_features, self.purifiers):
            scores = purifier.detector(feat)
            all_scores.append(scores)
        
        # Average scores across levels
        avg_score = torch.mean(torch.stack(all_scores), dim=0).squeeze()
        
        # Use threshold from first purifier (all have same threshold)
        return avg_score > self.purifiers[0].threshold


# Test function
if __name__ == "__main__":
    # Test single purifier
    purifier = SelectivePurificationModule(channels=256, threshold=0.5)
    x = torch.randn(4, 256, 8, 8)
    
    # Add synthetic "adversarial" perturbation to half the batch
    x[:2] += torch.randn_like(x[:2]) * 0.5
    
    purified, scores = purifier(x, return_scores=True)
    print(f"Input: {x.shape}, Purified: {purified.shape}")
    print(f"Anomaly scores: {scores.squeeze()}")
    print(f"Detected as adversarial: {purifier.is_adversarial(x)}")
    
    # Test multi-scale purifier
    ms_purifier = MultiScaleSelectivePurifier(fpn_channels=256, num_levels=4)
    fpn_feats = [torch.randn(4, 256, 8, 8) for _ in range(4)]
    purified_list, score_list = ms_purifier(fpn_feats, return_scores=True)
    
    print(f"\nMulti-scale purification:")
    print(f"Purified levels: {len(purified_list)}")
    print(f"Score levels: {len(score_list)}")
    print(f"Overall detection: {ms_purifier.detect_adversarial(fpn_feats)}")