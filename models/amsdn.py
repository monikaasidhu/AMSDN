import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.convnext_fpn import ConvNeXtFPN
from models.attention.adaptive_attention import MultiScaleAdaptiveAttention
from models.purification.selective_purifier import MultiScaleSelectivePurifier


class PredictionConsistencyVerifier(nn.Module):
    """Verifies prediction consistency across multiple forward passes"""
    
    def __init__(self, num_samples=5, consistency_threshold=0.8):
        super().__init__()
        self.num_samples = num_samples
        self.consistency_threshold = consistency_threshold
        
    def forward(self, x, model, noise_std=0.05):
        """
        Args:
            x: Input tensor [B, ...]
            model: Model to verify
            noise_std: Std of Gaussian noise for sampling
        Returns:
            is_consistent: [B] boolean tensor
            avg_logits: [B, num_classes] average logits
        """
        model.eval()
        all_logits = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                # Add small Gaussian noise
                noised = x + torch.randn_like(x) * noise_std
                logits = model(noised)
                all_logits.append(F.softmax(logits, dim=1))
        
        # Stack predictions
        all_probs = torch.stack(all_logits, dim=0)  # [num_samples, B, num_classes]
        
        # Average probabilities
        avg_probs = all_probs.mean(dim=0)  # [B, num_classes]
        
        # Compute prediction variance (consistency metric)
        pred_classes = all_probs.argmax(dim=2)  # [num_samples, B]
        mode_pred = torch.mode(pred_classes, dim=0)[0]  # [B]
        
        # Consistency: fraction of samples agreeing with mode
        consistency = (pred_classes == mode_pred.unsqueeze(0)).float().mean(dim=0)
        is_consistent = consistency >= self.consistency_threshold
        
        return is_consistent, avg_probs


class AMSDN(nn.Module):
    """
    Adaptive Multi-Scale Defense Network
    
    Pipeline:
    1. ConvNeXt-Tiny + FPN → Multi-scale features
    2. Adaptive Attention → Attended features
    3. Selective Purification → Cleaned features
    4. Prediction Consistency → Verification
    5. Classification → Final output
    """
    
    def __init__(self, num_classes=10, pretrained=True, 
                 purification_threshold=0.5, consistency_samples=5):
        super().__init__()
        
        # Stage 1: Backbone with FPN
        self.backbone = ConvNeXtFPN(num_classes=num_classes, pretrained=pretrained)
        
        # Stage 2: Adaptive Attention
        self.attention = MultiScaleAdaptiveAttention(fpn_channels=256, num_levels=4)
        
        # Stage 3: Selective Purification
        self.purifier = MultiScaleSelectivePurifier(
            fpn_channels=256, 
            num_levels=4, 
            threshold=purification_threshold
        )
        
        # Stage 4: Consistency Verifier
        self.verifier = PredictionConsistencyVerifier(
            num_samples=consistency_samples
        )
        
        # Classification head (operates on purified features)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward_features(self, x):
        """Extract and process multi-scale features"""
        # Stage 1: Multi-scale feature extraction
        fpn_features = self.backbone(x, return_features=True)
        
        # Stage 2: Adaptive attention
        attended_features, attention_maps = self.attention(fpn_features)
        
        # Stage 3: Selective purification
        purified_features, anomaly_scores = self.purifier(
            attended_features, 
            return_scores=True
        )
        
        return purified_features, anomaly_scores, attention_maps
    
    def forward(self, x, return_detailed=False, enable_verification=False):
        """
        Args:
            x: Input images [B, 3, H, W]
            return_detailed: If True, return all intermediate outputs
            enable_verification: If True, apply consistency verification
        Returns:
            If return_detailed:
                dict with logits, anomaly_scores, attention_maps, is_adversarial
            Else:
                logits only
        """
        # Extract and process features
        purified_features, anomaly_scores, attention_maps = self.forward_features(x)
        
        # Aggregate multi-scale features for classification
        pooled_features = []
        for feat in purified_features:
            pooled = self.global_pool(feat).flatten(1)
            pooled_features.append(pooled)
        
        combined = torch.cat(pooled_features, dim=1)
        logits = self.classifier(combined)
        
        # Stage 4: Consistency verification (optional, expensive)
        if enable_verification:
            is_consistent, verified_probs = self.verifier(x, self)
        else:
            is_consistent = None
            verified_probs = None
        
        # Adversarial detection (average anomaly scores across levels)
        avg_anomaly = torch.mean(torch.stack(anomaly_scores), dim=0).squeeze()
        is_adversarial = avg_anomaly > self.purifier.purifiers[0].threshold
        
        if return_detailed:
            return {
                'logits': logits,
                'anomaly_scores': anomaly_scores,
                'avg_anomaly_score': avg_anomaly,
                'is_adversarial': is_adversarial,
                'attention_maps': attention_maps,
                'is_consistent': is_consistent,
                'verified_probs': verified_probs
            }
        
        return logits
    
    def detect_and_reject(self, x, rejection_threshold=0.5):
        """
        Detect adversarial inputs and reject if necessary
        Args:
            x: Input images [B, 3, H, W]
            rejection_threshold: Anomaly score threshold
        Returns:
            logits: [B, num_classes] (or None for rejected samples)
            is_rejected: [B] boolean tensor
        """
        outputs = self.forward(x, return_detailed=True)
        
        is_rejected = outputs['avg_anomaly_score'] > rejection_threshold
        logits = outputs['logits']
        
        # Zero out logits for rejected samples
        logits[is_rejected] = float('-inf')
        
        return logits, is_rejected


class AMSDNWithSSRT(nn.Module):
    """
    AMSDN with Self-Supervised Robustness Training support
    Adds reconstruction head for pretraining
    """
    
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        
        self.amsdn = AMSDN(num_classes=num_classes, pretrained=pretrained)
        
        # Reconstruction head for SSRT
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(256 * 4, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x, mode='classify'):
        """
        Args:
            x: Input images
            mode: 'classify' or 'reconstruct'
        """
        if mode == 'classify':
            return self.amsdn(x)
        
        elif mode == 'reconstruct':
            # Get purified features
            purified_features, _, _ = self.amsdn.forward_features(x)
            
            # Upsample all features to same spatial size
            target_size = x.shape[-2:]
            upsampled = []
            for feat in purified_features:
                up = F.interpolate(feat, size=target_size, 
                                 mode='bilinear', align_corners=False)
                upsampled.append(up)
            
            # Concatenate and reconstruct
            concat = torch.cat(upsampled, dim=1)
            reconstructed = self.reconstruction_head(concat)
            
            return reconstructed


# Test function
if __name__ == "__main__":
    # Test AMSDN
    model = AMSDN(num_classes=10, pretrained=False)
    x = torch.randn(4, 3, 32, 32)
    
    print("=== Basic Forward Pass ===")
    logits = model(x)
    print(f"Logits: {logits.shape}")
    
    print("\n=== Detailed Output ===")
    outputs = model(x, return_detailed=True)
    print(f"Logits: {outputs['logits'].shape}")
    print(f"Anomaly scores (per level): {len(outputs['anomaly_scores'])}")
    print(f"Average anomaly: {outputs['avg_anomaly_score']}")
    print(f"Is adversarial: {outputs['is_adversarial']}")
    
    print("\n=== Detection and Rejection ===")
    logits_filtered, rejected = model.detect_and_reject(x)
    print(f"Rejected samples: {rejected.sum().item()}/{len(rejected)}")
    
    print("\n=== SSRT Model ===")
    ssrt_model = AMSDNWithSSRT(num_classes=10, pretrained=False)
    
    # Classification
    logits = ssrt_model(x, mode='classify')
    print(f"Classification logits: {logits.shape}")
    
    # Reconstruction
    reconstructed = ssrt_model(x, mode='reconstruct')
    print(f"Reconstructed: {reconstructed.shape}")