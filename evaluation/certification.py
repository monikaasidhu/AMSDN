"""
Randomized Smoothing Certification for AMSDN
Stage 6 of AMSDN pipeline
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from scipy.stats import norm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.amsdn import AMSDN
from data.cifar10 import CIFAR10DataModule


class RandomizedSmoothing:
    """
    Randomized Smoothing for certified robustness
    Based on Cohen et al. "Certified Adversarial Robustness via Randomized Smoothing"
    """
    
    def __init__(self, model, num_classes=10, sigma=0.25, device='cuda'):
        """
        Args:
            model: Base classifier
            num_classes: Number of classes
            sigma: Std of Gaussian noise for smoothing
            device: Device to run on
        """
        self.model = model.to(device)
        self.num_classes = num_classes
        self.sigma = sigma
        self.device = device
        self.model.eval()
    
    def predict_with_noise(self, x, num_samples=100):
        """
        Predict using Monte Carlo sampling with Gaussian noise
        Args:
            x: Input image [B, C, H, W]
            num_samples: Number of noise samples
        Returns:
            predictions: [B] class predictions
            class_counts: [B, num_classes] vote counts
        """
        B = x.size(0)
        class_counts = torch.zeros(B, self.num_classes, device=self.device)
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Add Gaussian noise
                noise = torch.randn_like(x) * self.sigma
                noised = x + noise
                
                # Predict
                outputs = self.model(noised)
                predictions = outputs.argmax(dim=1)
                
                # Accumulate votes
                for i in range(B):
                    class_counts[i, predictions[i]] += 1
        
        # Most voted class
        final_predictions = class_counts.argmax(dim=1)
        
        return final_predictions, class_counts
    
    def certify(self, x, n_samples_selection=100, n_samples_certification=10000, alpha=0.001):
        """
        Certify robustness for a single image
        Args:
            x: Input image [1, C, H, W]
            n_samples_selection: Samples for class selection
            n_samples_certification: Samples for certification
            alpha: Confidence level (1-alpha confidence)
        Returns:
            certified_class: Predicted class or -1 if abstain
            radius: Certified L2 radius or 0 if abstain
        """
        # Step 1: Select top class with smaller sample size
        _, counts_selection = self.predict_with_noise(x, n_samples_selection)
        top_class = counts_selection.argmax(dim=1).item()
        
        # Step 2: Certify with larger sample size
        _, counts_cert = self.predict_with_noise(x, n_samples_certification)
        
        # Count votes for top class
        n_A = counts_cert[0, top_class].item()
        
        # Confidence interval (Clopper-Pearson)
        p_A_lower = self._lower_confidence_bound(n_A, n_samples_certification, alpha)
        
        # Check if top class is certified
        if p_A_lower > 0.5:
            # Compute certified radius
            radius = self.sigma * norm.ppf(p_A_lower)
            return top_class, radius
        else:
            # Abstain (not certified)
            return -1, 0.0
    
    def _lower_confidence_bound(self, n_success, n_total, alpha):
        """
        Compute lower bound of confidence interval using Clopper-Pearson
        Args:
            n_success: Number of successes
            n_total: Total trials
            alpha: Confidence level
        Returns:
            Lower bound of binomial proportion
        """
        from scipy.stats import beta
        return beta.ppf(alpha, n_success, n_total - n_success + 1)
    
    def predict_batch(self, x, num_samples=100):
        """
        Predict for a batch (without certification)
        Args:
            x: Batch of images [B, C, H, W]
            num_samples: Number of Monte Carlo samples
        Returns:
            predictions: [B]
            probabilities: [B, num_classes]
        """
        predictions, class_counts = self.predict_with_noise(x, num_samples)
        
        # Convert counts to probabilities
        probabilities = class_counts / num_samples
        
        return predictions, probabilities


class CertificationEvaluator:
    """Evaluate certified robustness of AMSDN"""
    
    def __init__(self, model, sigma=0.25, device='cuda'):
        self.smoothed_model = RandomizedSmoothing(
            model, num_classes=10, sigma=sigma, device=device
        )
        self.device = device
    
    def evaluate_certified_accuracy(self, test_loader, max_samples=500, 
                                   n_samples=1000, alpha=0.001):
        """
        Evaluate certified accuracy at different radii
        Args:
            test_loader: Test data loader
            max_samples: Maximum number of samples to certify
            n_samples: Number of Monte Carlo samples per image
            alpha: Confidence level
        Returns:
            Dictionary with certification results
        """
        print("\n" + "=" * 60)
        print("Randomized Smoothing Certification")
        print("=" * 60)
        print(f"Sigma: {self.smoothed_model.sigma}")
        print(f"Samples per image: {n_samples}")
        print(f"Confidence level: {1-alpha:.3f}")
        print(f"Max samples: {max_samples}")
        
        certified_classes = []
        certified_radii = []
        true_labels = []
        
        sample_count = 0
        
        for images, labels in tqdm(test_loader, desc='Certifying'):
            if sample_count >= max_samples:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Certify each image individually
            for img, label in zip(images, labels):
                if sample_count >= max_samples:
                    break
                
                img_batch = img.unsqueeze(0)
                
                # Certify
                cert_class, radius = self.smoothed_model.certify(
                    img_batch, 
                    n_samples_selection=100,
                    n_samples_certification=n_samples,
                    alpha=alpha
                )
                
                certified_classes.append(cert_class)
                certified_radii.append(radius)
                true_labels.append(label.item())
                
                sample_count += 1
        
        # Compute metrics
        certified_classes = np.array(certified_classes)
        certified_radii = np.array(certified_radii)
        true_labels = np.array(true_labels)
        
        # Clean accuracy (radius > 0 and correct prediction)
        correct_predictions = (certified_classes == true_labels)
        clean_acc = correct_predictions.mean() * 100
        
        # Certified accuracy at different radii
        radii_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        certified_accs = {}
        
        for r in radii_levels:
            # Images certified at radius r with correct prediction
            certified_at_r = (certified_radii >= r) & correct_predictions
            cert_acc = certified_at_r.mean() * 100
            certified_accs[f'radius_{r}'] = cert_acc
        
        # Abstention rate (couldn't certify)
        abstain_rate = (certified_classes == -1).mean() * 100
        
        results = {
            'clean_accuracy': clean_acc,
            'certified_accuracies': certified_accs,
            'abstain_rate': abstain_rate,
            'avg_certified_radius': certified_radii[certified_radii > 0].mean() if (certified_radii > 0).any() else 0,
            'median_certified_radius': np.median(certified_radii[certified_radii > 0]) if (certified_radii > 0).any() else 0
        }
        
        return results
    
    def print_results(self, results):
        """Pretty print certification results"""
        print("\n" + "=" * 60)
        print("CERTIFICATION RESULTS")
        print("=" * 60)
        print(f"Clean Accuracy: {results['clean_accuracy']:.2f}%")
        print(f"Abstention Rate: {results['abstain_rate']:.2f}%")
        print(f"\nCertified Accuracy at Different Radii:")
        for radius_key, acc in results['certified_accuracies'].items():
            radius = float(radius_key.split('_')[1])
            print(f"  L2 radius {radius:.2f}: {acc:.2f}%")
        print(f"\nAverage Certified Radius: {results['avg_certified_radius']:.4f}")
        print(f"Median Certified Radius: {results['median_certified_radius']:.4f}")
        print("=" * 60)
    
    def compare_smoothed_vs_unsmoothed(self, test_loader, max_batches=10):
        """
        Compare performance of smoothed vs unsmoothed model
        """
        print("\n" + "=" * 60)
        print("Smoothed vs Unsmoothed Comparison")
        print("=" * 60)
        
        correct_smoothed = 0
        correct_unsmoothed = 0
        total = 0
        
        batch_count = 0
        
        for images, labels in tqdm(test_loader, desc='Comparing'):
            if batch_count >= max_batches:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Unsmoothed prediction
            with torch.no_grad():
                outputs_unsmoothed = self.smoothed_model.model(images)
                preds_unsmoothed = outputs_unsmoothed.argmax(dim=1)
            
            # Smoothed prediction
            preds_smoothed, _ = self.smoothed_model.predict_batch(images, num_samples=100)
            
            # Accuracy
            correct_unsmoothed += (preds_unsmoothed == labels).sum().item()
            correct_smoothed += (preds_smoothed == labels).sum().item()
            total += labels.size(0)
            
            batch_count += 1
        
        acc_unsmoothed = 100 * correct_unsmoothed / total
        acc_smoothed = 100 * correct_smoothed / total
        
        print(f"Unsmoothed Accuracy: {acc_unsmoothed:.2f}%")
        print(f"Smoothed Accuracy: {acc_smoothed:.2f}%")
        print(f"Accuracy Drop: {acc_unsmoothed - acc_smoothed:.2f}%")
        print("=" * 60)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-10 test set...")
    data_module = CIFAR10DataModule(batch_size=32, num_workers=2)
    _, test_loader = data_module.get_loaders()
    
    # Load trained model
    print("Loading trained AMSDN...")
    model = AMSDN(num_classes=10, pretrained=False)
    
    checkpoint_path = './checkpoints/finetuned/amsdn_finetuned_best.pth'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = './checkpoints/adversarial/amsdn_best.pth'
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: No trained model found. Using random initialization.")
    
    # Certification evaluator
    evaluator = CertificationEvaluator(model, sigma=0.25, device=device)
    
    # Compare smoothed vs unsmoothed
    evaluator.compare_smoothed_vs_unsmoothed(test_loader, max_batches=10)
    
    # Run certification (this is slow, use small sample size for demo)
    print("\nRunning certification (this may take a while)...")
    results = evaluator.evaluate_certified_accuracy(
        test_loader,
        max_samples=100,  # Use more for real evaluation (e.g., 500-1000)
        n_samples=500,     # Use more for real certification (e.g., 10000)
        alpha=0.001
    )
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    import json
    os.makedirs('./results', exist_ok=True)
    with open('./results/certification_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\n✓ Results saved to ./results/certification_results.json")


if __name__ == "__main__":
    main()