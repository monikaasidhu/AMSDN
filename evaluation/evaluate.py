"""
Comprehensive Evaluation of AMSDN
Tests against multiple attacks and metrics
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.amsdn import AMSDN
from data.cifar10 import CIFAR10DataModule
from attacks.patch_attacks import AdversarialPatch, AdaptivePatchwithBPDA
from attacks.pixel_attacks import FewPixelAttack
from training.adversarial_train import PGDAttack


class AMSDNEvaluator:
    """Comprehensive evaluation suite for AMSDN"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Initialize attacks
        self.attacks = {
            'PGD-8': PGDAttack(epsilon=8/255, alpha=2/255, num_steps=20),
            'PGD-16': PGDAttack(epsilon=16/255, alpha=2/255, num_steps=20),
            'Patch-4': AdversarialPatch(patch_size=4, epsilon=0.3, num_steps=100),
            'Patch-8': AdversarialPatch(patch_size=8, epsilon=0.5, num_steps=100),
            'Pixel-5': FewPixelAttack(num_pixels=5, epsilon=0.5, num_iterations=50),
            'Pixel-10': FewPixelAttack(num_pixels=10, epsilon=0.5, num_iterations=50),
        }
    
    def evaluate_clean(self, test_loader):
        """Evaluate clean accuracy"""
        correct = 0
        total = 0
        inference_times = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Clean evaluation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                outputs = self.model(images)
                inference_times.append(time.time() - start_time)
                
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100 * correct / total
        avg_time = sum(inference_times) / len(inference_times)
        
        return {
            'accuracy': accuracy,
            'avg_inference_time': avg_time,
            'throughput': len(images) / avg_time
        }
    
    def evaluate_attack(self, test_loader, attack_name, attack, max_batches=None):
        """
        Evaluate against specific attack
        Returns:
            - Robust accuracy
            - Attack success rate (ASR)
            - Detection rate
        """
        correct = 0
        total = 0
        detected = 0
        
        batch_count = 0
        
        for images, labels in tqdm(test_loader, desc=f'{attack_name} evaluation'):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Generate adversarial examples
            if attack_name.startswith('PGD'):
                adv_images = attack.generate(self.model, images, labels)
            elif attack_name.startswith('Patch'):
                adv_images = attack.apply(images, self.model, labels, optimize=True)
            else:  # Pixel attacks
                adv_images = attack.attack(images, self.model, labels)
            
            # Evaluate with detection
            with torch.no_grad():
                outputs = self.model(adv_images, return_detailed=True)
                preds = outputs['logits'].argmax(dim=1)
                is_detected = outputs['is_adversarial']
                
                correct += (preds == labels).sum().item()
                detected += is_detected.sum().item()
                total += labels.size(0)
            
            batch_count += 1
            if max_batches and batch_count >= max_batches:
                break
        
        robust_accuracy = 100 * correct / total
        asr = 100 * (1 - correct / total)  # Attack success rate
        detection_rate = 100 * detected / total
        
        return {
            'robust_accuracy': robust_accuracy,
            'attack_success_rate': asr,
            'detection_rate': detection_rate
        }
    
    def evaluate_all_attacks(self, test_loader, max_batches=20):
        """Evaluate against all attacks"""
        results = {}
        
        print("\n" + "=" * 60)
        print("Evaluating AMSDN")
        print("=" * 60)
        
        # Clean accuracy
        print("\n1. Clean Accuracy")
        clean_results = self.evaluate_clean(test_loader)
        results['clean'] = clean_results
        print(f"   Accuracy: {clean_results['accuracy']:.2f}%")
        print(f"   Avg inference time: {clean_results['avg_inference_time']*1000:.2f}ms")
        print(f"   Throughput: {clean_results['throughput']:.1f} imgs/sec")
        
        # Evaluate each attack
        print("\n2. Adversarial Robustness")
        for attack_name, attack in self.attacks.items():
            print(f"\n   {attack_name}:")
            attack_results = self.evaluate_attack(
                test_loader, attack_name, attack, max_batches
            )
            results[attack_name] = attack_results
            
            print(f"     Robust Accuracy: {attack_results['robust_accuracy']:.2f}%")
            print(f"     ASR: {attack_results['attack_success_rate']:.2f}%")
            print(f"     Detection Rate: {attack_results['detection_rate']:.2f}%")
        
        return results
    
    def evaluate_adaptive_attack(self, test_loader, max_batches=10):
        """Evaluate against adaptive BPDA-style attack"""
        print("\n3. Adaptive Attack (BPDA)")
        
        adaptive_attack = AdaptivePatchwithBPDA(patch_size=8, epsilon=0.5)
        
        correct = 0
        total = 0
        detected = 0
        
        batch_count = 0
        
        for images, labels in tqdm(test_loader, desc='Adaptive attack'):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Generate adaptive adversarial examples
            adv_images = adaptive_attack.attack(images, self.model, labels)
            
            # Evaluate
            with torch.no_grad():
                outputs = self.model(adv_images, return_detailed=True)
                preds = outputs['logits'].argmax(dim=1)
                is_detected = outputs['is_adversarial']
                
                correct += (preds == labels).sum().item()
                detected += is_detected.sum().item()
                total += labels.size(0)
            
            batch_count += 1
            if batch_count >= max_batches:
                break
        
        robust_acc = 100 * correct / total
        asr = 100 * (1 - correct / total)
        detection_rate = 100 * detected / total
        
        print(f"   Robust Accuracy: {robust_acc:.2f}%")
        print(f"   ASR: {asr:.2f}%")
        print(f"   Detection Rate: {detection_rate:.2f}%")
        
        return {
            'robust_accuracy': robust_acc,
            'attack_success_rate': asr,
            'detection_rate': detection_rate
        }
    
    def save_results(self, results, save_path='./results/evaluation_results.json'):
        """Save evaluation results to JSON"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n✓ Results saved to {save_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-10 test set...")
    data_module = CIFAR10DataModule(batch_size=64, num_workers=2)
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
    
    # Evaluate
    evaluator = AMSDNEvaluator(model, device)
    
    # Standard attacks
    results = evaluator.evaluate_all_attacks(test_loader, max_batches=20)
    
    # Adaptive attack
    adaptive_results = evaluator.evaluate_adaptive_attack(test_loader, max_batches=10)
    results['Adaptive-BPDA'] = adaptive_results
    
    # Save results
    evaluator.save_results(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Clean Accuracy: {results['clean']['accuracy']:.2f}%")
    print(f"Average Robust Accuracy: {sum([v['robust_accuracy'] for k, v in results.items() if k != 'clean']) / (len(results) - 1):.2f}%")
    print(f"Average Detection Rate: {sum([v['detection_rate'] for k, v in results.items() if k != 'clean']) / (len(results) - 1):.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()