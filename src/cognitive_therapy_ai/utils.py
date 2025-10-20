"""
Utility functions for the cognitive therapy AI framework.
"""

import torch
import numpy as np
import random
import logging
import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("cognitive_therapy_ai")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_output_dirs(base_dir: str, experiment_name: str) -> Dict[str, str]:
    """Create output directories for experiment results."""
    dirs = {
        'base': os.path.join(base_dir, experiment_name),
        'checkpoints': os.path.join(base_dir, experiment_name, 'checkpoints'),
        'logs': os.path.join(base_dir, experiment_name, 'logs'),
        'plots': os.path.join(base_dir, experiment_name, 'plots'),
        'results': os.path.join(base_dir, experiment_name, 'results')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save experiment results to file."""
    import pickle
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)


def load_results(filepath: str) -> Dict[str, Any]:
    """Load experiment results from file."""
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)


class MetricsTracker:
    """Track and visualize training metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def add_metric(self, name: str, value: float, step: int = None):
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = {'values': [], 'steps': []}
        
        self.metrics[name]['values'].append(value)
        if step is not None:
            self.metrics[name]['steps'].append(step)
        else:
            self.metrics[name]['steps'].append(len(self.metrics[name]['values']))
    
    def get_metric(self, name: str) -> Dict[str, List]:
        """Get metric history."""
        return self.metrics.get(name, {'values': [], 'steps': []})
    
    def plot_metrics(self, metrics_to_plot: List[str] = None, save_path: str = None):
        """Plot training metrics."""
        if metrics_to_plot is None:
            metrics_to_plot = list(self.metrics.keys())
        
        n_metrics = len(metrics_to_plot)
        if n_metrics == 0:
            return
        
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4*n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric_name in enumerate(metrics_to_plot):
            if metric_name in self.metrics:
                metric_data = self.metrics[metric_name]
                axes[i].plot(metric_data['steps'], metric_data['values'])
                axes[i].set_title(f'{metric_name}')
                axes[i].set_xlabel('Step')
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def calculate_moving_average(values: List[float], window_size: int = 10) -> List[float]:
    """Calculate moving average of values."""
    if len(values) < window_size:
        return values
    
    moving_avg = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size + 1)
        avg = sum(values[start_idx:i+1]) / (i - start_idx + 1)
        moving_avg.append(avg)
    
    return moving_avg