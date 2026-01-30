"""Utility classes for monitoring convergence and accumulating losses."""
import torch


class ConvergenceMonitor:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait_count = 0
    
    def update(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait_count = 0
        else:
            self.wait_count += 1
            
        if self.wait_count >= self.patience:
            return True
            
        return False


class NMSELossAccumulator:
    def __init__(self):
        self._total_nmse = 0.0
        self._num_samples = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update with a batch of predictions and targets"""
        assert predictions.shape == targets.shape
        
        batch_size = predictions.shape[0]
        mse = torch.sum((predictions - targets) ** 2, dim=(-1, -2))
        target_power = torch.sum(targets ** 2, dim=(-1, -2))
        nmse_db = 10 * torch.log10(mse / target_power)
        self._total_nmse += torch.sum(nmse_db).item()
        self._num_samples += batch_size
    
    def get_total_nmse_loss(self) -> float:
        """Compute final NMSE loss in dB"""
        if self._num_samples == 0:
            return 0.0
        return self._total_nmse / self._num_samples
    
    def reset(self):
        """Reset accumulator"""
        self._total_nmse = 0.0
        self._num_samples = 0


class LogLikelihood:
    def __init__(self):
        self._total_log_likelihood = 0.0
        self._num_samples = 0
    
    def update(
        self,
        mu: torch.Tensor,
        cov: torch.Tensor,
        targets: torch.Tensor
    ):
        B, T, D = mu.shape
        eye = torch.eye(D, device=cov.device, dtype=cov.dtype)
        cov_stable = cov + 1e-6 * eye.unsqueeze(0).unsqueeze(0)
        
        mu_flat = mu.view(B * T, D)
        cov_flat = cov_stable.view(B * T, D, D)
        x_flat = targets.view(B * T, D)
        
        dist = torch.distributions.MultivariateNormal(mu_flat, cov_flat)
        self._total_log_likelihood += torch.sum(dist.log_prob(x_flat)).item()
        self._num_samples += B * T

    def get_total_log_likelihood(self) -> float:
        """Compute final log likelihood"""
        if self._num_samples == 0:
            return 0.0
        return self._total_log_likelihood / self._num_samples
    
    def reset(self):
        """Reset accumulator"""
        self._total_log_likelihood = 0.0
        self._num_samples = 0