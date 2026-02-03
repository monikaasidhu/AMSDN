"""
Adversarial Training for AMSDN
Joint detection + purification + classification training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.amsdn import AMSDN
from data.cifar10 import CIFAR10DataModule


class PGDAttack:
    """PGD attack for adversarial training"""
    
    def __init__(self, epsilon=8/255, alpha=2/255, num_steps=10):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
    
    def generate(self, model, images, labels):
        """Generate PGD adversarial examples"""
        images_adv = images.clone().detach()
        
        for _ in range(self.num_steps):
            images_adv = images_adv.clone().detach().requires_grad_(True)
            
            outputs = model(images_adv)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            grad = torch.autograd.grad(loss, images_adv)[0]
            
            # Update
            images_adv = images_adv.detach() + self.alpha * grad.sign()
            
            # Project back to epsilon ball
            delta = torch.clamp(images_adv - images, -self.epsilon, self.epsilon)
            images_adv = torch.clamp(images + delta, -2, 2)  # Normalized range
        
        return images_adv.detach()


class AdversarialTrainer:
    """Adversarial training for AMSDN"""
    
    def __init__(self, model, device='cuda', lr=1e-4, save_dir='./checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Attack for training
        self.attack = PGDAttack(epsilon=8/255, alpha=2/255, num_steps=10)
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[50, 75, 90], gamma=0.1
        )
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Logging
        self.writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
    
    def compute_loss(self, outputs, labels, is_adversarial_gt):
        """
        Compute joint loss:
        - Classification loss
        - Detection loss
        """
        logits = outputs['logits']
        avg_anomaly = outputs['avg_anomaly_score']
        
        # Classification loss
        cls_loss = self.ce_loss(logits, labels)
        
        # Detection loss (binary: clean=0, adversarial=1)
        det_loss = self.bce_loss(
            avg_anomaly.squeeze(), 
            is_adversarial_gt.float()
        )
        
        # Combined loss
        total_loss = cls_loss + 0.5 * det_loss
        
        return total_loss, cls_loss, det_loss
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_cls = 0
        total_det = 0
        correct_clean = 0
        correct_adv = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)
            
            # Split batch: half clean, half adversarial
            split = batch_size // 2
            
            # Clean samples
            clean_images = images[:split]
            clean_labels = labels[:split]
            
            # Generate adversarial samples
            adv_images = self.attack.generate(
                self.model, 
                images[split:], 
                labels[split:]
            )
            adv_labels = labels[split:]
            
            # Combine
            combined_images = torch.cat([clean_images, adv_images], dim=0)
            combined_labels = torch.cat([clean_labels, adv_labels], dim=0)
            is_adversarial = torch.cat([
                torch.zeros(split, device=self.device),
                torch.ones(batch_size - split, device=self.device)
            ])
            
            # Forward pass
            outputs = self.model(combined_images, return_detailed=True)
            
            # Compute loss
            loss, cls_loss, det_loss = self.compute_loss(
                outputs, combined_labels, is_adversarial
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            preds = outputs['logits'].argmax(dim=1)
            correct_clean += (preds[:split] == clean_labels).sum().item()
            correct_adv += (preds[split:] == adv_labels).sum().item()
            total_samples += batch_size
            
            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_det += det_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}',
                'det': f'{det_loss.item():.4f}',
                'acc_c': f'{100*correct_clean/(split*(batch_idx+1)):.1f}%',
                'acc_a': f'{100*correct_adv/((batch_size-split)*(batch_idx+1)):.1f}%'
            })
            
            # Log to tensorboard
            step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), step)
            self.writer.add_scalar('Train/Classification', cls_loss.item(), step)
            self.writer.add_scalar('Train/Detection', det_loss.item(), step)
        
        # Average metrics
        avg_loss = total_loss / len(train_loader)
        avg_cls = total_cls / len(train_loader)
        avg_det = total_det / len(train_loader)
        clean_acc = 100 * correct_clean / (total_samples // 2)
        adv_acc = 100 * correct_adv / (total_samples - total_samples // 2)
        
        return avg_loss, avg_cls, avg_det, clean_acc, adv_acc
    
    def evaluate(self, test_loader, attack_epsilon=8/255):
        self.model.eval()

        correct_clean = 0
        correct_adv = 0
        detected_adv = 0
        total = 0

        attack = PGDAttack(epsilon=attack_epsilon, alpha=2/255, num_steps=20)

        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # ---- CLEAN PREDICTION (no gradients needed) ----
            with torch.no_grad():
                outputs_clean = self.model(images, return_detailed=True)
                preds_clean = outputs_clean['logits'].argmax(dim=1)
                correct_clean += (preds_clean == labels).sum().item()

            # ---- ADVERSARIAL GENERATION (gradients REQUIRED) ----
            images_adv = images.clone().detach().requires_grad_(True)
            adv_images = attack.generate(self.model, images_adv, labels)

            # ---- ADVERSARIAL PREDICTION (no gradients needed) ----
            with torch.no_grad():
                outputs_adv = self.model(adv_images, return_detailed=True)
                preds_adv = outputs_adv['logits'].argmax(dim=1)
                correct_adv += (preds_adv == labels).sum().item()
                detected_adv += outputs_adv['is_adversarial'].sum().item()

            total += labels.size(0)

        clean_acc = 100 * correct_clean / total
        adv_acc = 100 * correct_adv / total
        detection_rate = 100 * detected_adv / total

        return clean_acc, adv_acc, detection_rate

    def train(self, train_loader, test_loader, num_epochs=100):
        """Full training loop"""
        print("=" * 60)
        print("Starting Adversarial Training")
        print("=" * 60)
        
        best_adv_acc = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            avg_loss, avg_cls, avg_det, train_clean, train_adv = self.train_epoch(
                train_loader, epoch
            )
            
            # Evaluate every 5 epochs
            if epoch % 5 == 0:
                clean_acc, adv_acc, det_rate = self.evaluate(test_loader)
                
                print(f"\nEpoch {epoch}/{num_epochs}:")
                print(f"  Train - Loss: {avg_loss:.4f}, Clean: {train_clean:.2f}%, Adv: {train_adv:.2f}%")
                print(f"  Test  - Clean: {clean_acc:.2f}%, Adv: {adv_acc:.2f}%, Detection: {det_rate:.2f}%")
                
                self.writer.add_scalar('Test/CleanAccuracy', clean_acc, epoch)
                self.writer.add_scalar('Test/AdversarialAccuracy', adv_acc, epoch)
                self.writer.add_scalar('Test/DetectionRate', det_rate, epoch)
                
                # Save best model
                if adv_acc > best_adv_acc:
                    best_adv_acc = adv_acc
                    checkpoint_path = os.path.join(self.save_dir, 'amsdn_best.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'adv_accuracy': adv_acc,
                    }, checkpoint_path)
                    print(f"  ✓ Saved best model (adv acc: {best_adv_acc:.2f}%)")
            
            # Update scheduler
            self.scheduler.step()
            
            # Save periodic checkpoint
            if epoch % 20 == 0:
                checkpoint_path = os.path.join(self.save_dir, f'amsdn_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, checkpoint_path)
        
        self.writer.close()
        print("\n" + "=" * 60)
        print("Adversarial Training Complete!")
        print(f"Best Adversarial Accuracy: {best_adv_acc:.2f}%")
        print("=" * 60)


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("Loading CIFAR-10 dataset...")
    data_module = CIFAR10DataModule(batch_size=128, num_workers=2)
    train_loader, test_loader = data_module.get_loaders()
    
    # Model
    print("Initializing AMSDN...")
    model = AMSDN(num_classes=10, pretrained=True)
    
    # Load SSRT pretrained weights if available
    ssrt_checkpoint = './checkpoints/ssrt/ssrt_best.pth'
    if os.path.exists(ssrt_checkpoint):
        print("Loading SSRT pretrained weights...")
        checkpoint = torch.load(ssrt_checkpoint, map_location=device)
        # Load only compatible weights
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                          if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} layers")
    
    # Trainer
    trainer = AdversarialTrainer(
        model=model,
        device=device,
        lr=1e-4,
        save_dir='./checkpoints/adversarial'
    )
    
    # Train
    trainer.train(train_loader, test_loader, num_epochs=100)


if __name__ == "__main__":
    main()