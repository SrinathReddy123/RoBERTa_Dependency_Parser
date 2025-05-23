"""
Dependency Parser Package
A neural dependency parser implementation using PyTorch and RoBERTa.
"""

import os
import logging
import torch
from pathlib import Path

# Package version
__version__ = "1.0.0"

# Setup logging configuration
def setup_logging(log_dir: str = "logs"):
    """Setup logging configuration for the package."""
    try:
        # Convert to absolute path
        log_dir = Path(log_dir).absolute()
        print(f"Setting up logging in directory: {log_dir}")  # Debug output
        
        # Create logs directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # Create file handlers for each module
        modules = ['model', 'trainer', 'data_utils', 'preprocess']
        for module in modules:
            log_file = log_dir / f"{module}.log"
            try:
                print(f"Creating log file: {log_file}")  # Debug output
                file_handler = logging.FileHandler(log_file, mode='w')
                file_handler.setFormatter(file_formatter)
                logger = logging.getLogger(module)
                logger.setLevel(logging.INFO)
                
                # Clear existing handlers for this logger
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                    handler.close()
                
                logger.addHandler(file_handler)
                logger.propagate = False
                logger.info(f"Successfully initialized logging for {module}")
            except Exception as e:
                print(f"Failed to setup logging for {module}: {str(e)}")
                raise
        
        print("Logging setup completed successfully")
    except Exception as e:
        print(f"CRITICAL ERROR in logging setup: {str(e)}")
        raise

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data" / "ewt"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
PLOT_DIR = ROOT_DIR / "plots"
LOG_DIR = ROOT_DIR / "logs"

# Create necessary directories
for directory in [DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, PLOT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# GPU Configuration
def setup_gpu():
    """Configure GPU settings for optimal performance."""
    if torch.cuda.is_available():
        # Set default tensor type to cuda
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        
        # Enable cuDNN benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        
        # Set memory growth to prevent OOM errors
        torch.cuda.empty_cache()
        
        return torch.device("cuda")
    return torch.device("cpu")

# Initialize logging and GPU
setup_logging(str(LOG_DIR))
DEVICE = setup_gpu()

# Import main components
from .model import (
    DependencyParser,
    DependencyParsingLoss,
    EnhancedDependencyParsingLoss,
    FocalLoss,
    LabelSmoothingCrossEntropy,
    DiceLoss,
    ContrastiveLoss
)

from .data_utils import (
    DependencyParsingDataset,
    get_data_loaders,
    load_label_vocab
)

from .trainer import (
    DependencyParserTrainer,
    MetricsTracker,
    hyperparameter_search,
    plot_loss_function_comparison
)

# Export main components
__all__ = [
    # Models
    'DependencyParser',
    'DependencyParsingLoss',
    'EnhancedDependencyParsingLoss',
    'FocalLoss',
    'LabelSmoothingCrossEntropy',
    'DiceLoss',
    'ContrastiveLoss',
    
    # Data utilities
    'DependencyParsingDataset',
    'get_data_loaders',
    'load_label_vocab',
    
    # Training utilities
    'DependencyParserTrainer',
    'MetricsTracker',
    'hyperparameter_search',
    'plot_loss_function_comparison',
    
    # Constants
    'DEVICE',
    'ROOT_DIR',
    'DATA_DIR',
    'PROCESSED_DATA_DIR',
    'CHECKPOINT_DIR',
    'PLOT_DIR',
    'LOG_DIR'
] 