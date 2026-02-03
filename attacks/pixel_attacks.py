"""
One-Pixel and Few-Pixel Attacks
Sparse adversarial perturbations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OnePixelAttack:
    """
    One-pixel attack using differential evolution
    Based on Su et al. "One pixel attack for fooling deep neural networks"
    """
    
    def __init__(self, pop_size=400, max_iterations=100):
        self.pop_size = pop_size
        self.max_iterations = max_iterations
    
    def differential_evolution(self, model, image, label):
        """
        Use differential evolution to find adversarial pixel
        Args:
            model: Target model
            image: Single image [C, H, W]
            label: True label (scalar)
        Returns:
            adversarial image [C, H, W]
        """
        device = image.device
        C, H, W = image.shape
        
        # Initialize population: (x, y, r, g, b)
        # x, y in [0, H-1], [0, W-1], rgb in [-2, 2] (normalized range)
        population = np.random.rand(self.pop_size, 5)
        population[:, 0] *= H  # x position
        population[:, 1] *= W  # y position
        population[:, 2:] = population[:, 2:] * 4 - 2  # rgb values
        
        best_solution = None
        best_fitness = float('-inf')
        
        model.eval()
        with torch.no_grad():
            for iteration in range(self.max_iterations):
                # Evaluate population
                fitness_scores = []
                
                for individual in population:
                    # Create perturbed image
                    perturbed = image.clone()
                    x, y = int(individual[0]), int(individual[1])
                    x = min(max(x, 0), H-1)
                    y = min(max(y, 0), W-1)
                    perturbed[:, x, y] = torch.tensor(individual[2:], device=device)
                    
                    # Evaluate
                    output = model(perturbed.unsqueeze(0))
                    probs = F.softmax(output, dim=1)[0]
                    
                    # Fitness: probability of wrong class
                    true_prob = probs[label].item()
                    fitness = 1 - true_prob
                    fitness_scores.append(fitness)
                    
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_solution = individual.copy()
                
                # Differential evolution update
                fitness_scores = np.array(fitness_scores)
                new_population = []
                
                for i in range(self.pop_size):
                    # Select three random individuals
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    a, b, c = population[indices]
                    
                    # Mutation
                    F = 0.8
                    mutant = a + F * (b - c)
                    
                    # Crossover
                    CR = 0.9
                    trial = population[i].copy()
                    for j in range(5):
                        if np.random.rand() < CR:
                            trial[j] = mutant[j]
                    
                    # Clip to valid ranges
                    trial[0] = np.clip(trial[0], 0, H-1)
                    trial[1] = np.clip(trial[1], 0, W-1)
                    trial[2:] = np.clip(trial[2:], -2, 2)
                    
                    # Selection
                    new_population.append(trial)
                
                population = np.array(new_population)
        
        # Apply best solution
        if best_solution is not None:
            adversarial = image.clone()
            x, y = int(best_solution[0]), int(best_solution[1])
            x = min(max(x, 0), H-1)
            y = min(max(y, 0), W-1)
            adversarial[:, x, y] = torch.tensor(best_solution[2:], device=device)
            return adversarial
        
        return image
    
    def attack(self, images, model, labels):
        """
        Attack batch of images
        Args:
            images: [B, C, H, W]
            model: Target model
            labels: [B]
        Returns:
            adversarial images [B, C, H, W]
        """
        adversarial_images = []
        
        for img, label in zip(images, labels):
            adv_img = self.differential_evolution(model, img, label.item())
            adversarial_images.append(adv_img)
        
        return torch.stack(adversarial_images)


class FewPixelAttack:
    """
    Few-pixel attack (extension of one-pixel)
    Perturb multiple pixels for stronger attack
    """
    
    def __init__(self, num_pixels=5, epsilon=0.5, num_iterations=50):
        self.num_pixels = num_pixels
        self.epsilon = epsilon
        self.num_iterations = num_iterations
    
    def select_pixels_greedy(self, model, image, label, num_pixels):
        """
        Greedily select most important pixels to perturb
        Based on gradient magnitude
        """
        device = image.device
        image_var = image.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = model(image_var.unsqueeze(0))
        loss = F.cross_entropy(output, label.unsqueeze(0))
        
        # Backward to get gradients
        loss.backward()
        
        # Get gradient magnitude per pixel (sum over channels)
        grad = image_var.grad
        grad_magnitude = grad.abs().sum(dim=0)  # [H, W]
        
        # Select top pixels
        H, W = grad_magnitude.shape
        flat_grad = grad_magnitude.flatten()
        _, top_indices = torch.topk(flat_grad, num_pixels)
        
        # Convert to (x, y) coordinates
        pixels = []
        for idx in top_indices:
            x = idx // W
            y = idx % W
            pixels.append((x.item(), y.item()))
        
        return pixels
    
    def optimize_pixel_values(self, model, image, label, pixel_positions):
        """
        Optimize values for selected pixels
        """
        device = image.device
        adversarial = image.clone()
        
        # Initialize pixel values
        pixel_values = torch.zeros(len(pixel_positions), 3, device=device)
        for i, (x, y) in enumerate(pixel_positions):
            pixel_values[i] = image[:, x, y]
        
        pixel_values.requires_grad = True
        optimizer = torch.optim.Adam([pixel_values], lr=0.01)
        
        for iteration in range(self.num_iterations):
            # Apply pixel values
            adv_image = adversarial.clone()
            for i, (x, y) in enumerate(pixel_positions):
                adv_image[:, x, y] = pixel_values[i]
            
            # Forward pass
            output = model(adv_image.unsqueeze(0))
            
            # Loss: maximize probability of wrong class
            loss = -F.cross_entropy(output, label.unsqueeze(0))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Project to epsilon ball around original
            with torch.no_grad():
                for i, (x, y) in enumerate(pixel_positions):
                    original_val = image[:, x, y]
                    pixel_values[i] = torch.clamp(
                        pixel_values[i],
                        original_val - self.epsilon,
                        original_val + self.epsilon
                    )
                    pixel_values[i] = torch.clamp(pixel_values[i], -2, 2)
        
        # Apply final values
        for i, (x, y) in enumerate(pixel_positions):
            adversarial[:, x, y] = pixel_values[i].detach()
        
        return adversarial
    
    def attack(self, images, model, labels):
        """
        Attack batch of images
        """
        model.eval()
        adversarial_images = []
        
        for img, label in zip(images, labels):
            # Select pixels
            pixels = self.select_pixels_greedy(model, img, label, self.num_pixels)
            
            # Optimize pixel values
            adv_img = self.optimize_pixel_values(model, img, label, pixels)
            adversarial_images.append(adv_img)
        
        return torch.stack(adversarial_images)


class SparseAttackEvaluator:
    """Evaluate sparse attacks with different sparsity levels"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def evaluate_sparsity_levels(self, images, labels, max_pixels=10):
        """
        Evaluate attack success rate vs number of perturbed pixels
        """
        results = {}
        
        for num_pixels in range(1, max_pixels + 1):
            attack = FewPixelAttack(num_pixels=num_pixels, epsilon=0.5)
            
            # Generate adversarial examples
            adv_images = attack.attack(images, self.model, labels)
            
            # Evaluate
            with torch.no_grad():
                clean_outputs = self.model(images)
                adv_outputs = self.model(adv_images)
                
                clean_preds = clean_outputs.argmax(dim=1)
                adv_preds = adv_outputs.argmax(dim=1)
                
                # Attack success: changed prediction
                success = (adv_preds != labels).float().mean().item()
                
                # Average perturbation
                l2_norm = (adv_images - images).pow(2).sum(dim=(1, 2, 3)).sqrt().mean().item()
                
                results[num_pixels] = {
                    'success_rate': success,
                    'l2_norm': l2_norm
                }
        
        return results


# Test
if __name__ == "__main__":
    print("Testing pixel attacks...")
    
    # Dummy data
    images = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 1])
    
    # Dummy model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    model.eval()
    
    # Test few-pixel attack
    print("\nFew-Pixel Attack:")
    few_pixel_attack = FewPixelAttack(num_pixels=5, epsilon=0.5)
    adv_images = few_pixel_attack.attack(images, model, labels)
    
    with torch.no_grad():
        clean_outputs = model(images)
        adv_outputs = model(adv_images)
        
        clean_preds = clean_outputs.argmax(dim=1)
        adv_preds = adv_outputs.argmax(dim=1)
        
        print(f"Clean predictions: {clean_preds}")
        print(f"Adversarial predictions: {adv_preds}")
        print(f"Labels: {labels}")
        print(f"Success rate: {(adv_preds != labels).float().mean():.2%}")
    
    # Compute perturbation statistics
    diff = (adv_images - images).abs()
    num_changed_pixels = (diff.sum(dim=1) > 0).float().sum(dim=(1, 2))
    print(f"Pixels changed per image: {num_changed_pixels}")