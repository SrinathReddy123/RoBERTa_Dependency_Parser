from logger_setup import setup_logging
setup_logging()

import os
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
import shutil
from model import DependencyParser, MultiTaskLoss, ArcMarginLoss
from data_utils import get_data_loaders

import json
import numpy as np

import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),  # Logs to file
        logging.StreamHandler()               # Logs to console
    ]
)

# Create a logger for this module
logger = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class ExperimentTracker:
    """Tracks experiment state and manages checkpoints"""
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        self.plots_dir = os.path.join(experiment_dir, "plots")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        self.current_loss_type = None
        self.current_hp_config = None
        self.best_metrics = defaultdict(dict)
        self.all_metrics = defaultdict(list)
        
    def init_experiment(self, loss_type, hp_config):
        """Initialize tracking for a new loss function experiment"""
        self.current_loss_type = loss_type
        self.current_hp_config = hp_config
        self.current_run_id = f"{loss_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def save_checkpoint(self, epoch, model, optimizer, metrics):
        """Save full experiment state"""
        checkpoint = {
            'epoch': epoch,
            'loss_type': self.current_loss_type,
            'hp_config': self.current_hp_config,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(
            checkpoint, 
            os.path.join(self.checkpoint_dir, f"{self.current_run_id}_epoch_{epoch}.pt")
        )
        
    def load_checkpoint(self, loss_type, hp_config):
        """Load the latest checkpoint for a specific experiment"""
        pattern = f"{loss_type}_*"
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith(loss_type)]
        if not checkpoints:
            return None
            
        # Find most recent checkpoint
        latest_checkpoint = sorted(checkpoints)[-1]
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, latest_checkpoint))
        
        # Verify it matches our current HP config
        if checkpoint['hp_config'] != hp_config:
            logger.warning("Loaded checkpoint has different hyperparameters!")
            
        return checkpoint

class EarlyStopper:
    """Enhanced early stopping with multiple metric tracking"""
    def __init__(self, patience=5, min_delta=0.001, metrics=('dev_uas', 'dev_las')):
        self.patience = patience
        self.min_delta = min_delta
        self.metrics = metrics
        self.best_scores = {metric: -np.inf for metric in metrics}
        self.counter = 0
        
    def update(self, current_metrics):
        """Returns True if training should stop"""
        should_stop = True
        for metric in self.metrics:
            if current_metrics[metric] > self.best_scores[metric] + self.min_delta:
                self.best_scores[metric] = current_metrics[metric]
                self.counter = 0
                should_stop = False
                break
                
        if should_stop:
            self.counter += 1
            
        return self.counter >= self.patience

class DependencyParserTrainer:
    def __init__(self, model, loss_fn, optimizer, device, label2id, id2label, config):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.label2id = label2id
        self.id2label = id2label
        self.config = config
        
        # Initialize trackers
        self.tracker = ExperimentTracker(config['log_dir'])
        self.early_stopper = EarlyStopper(
            patience=config.get('patience', 5),
            min_delta=config.get('min_delta', 0.001)
        )
        
        # Training state
        self.current_epoch = 0
        self.metrics_history = defaultdict(list)
        
        # Visualization
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
    def train(self, train_loader, dev_loader, max_epochs):
        """Full training loop with checkpointing and early stopping"""
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Evaluation
            dev_metrics = self.evaluate(dev_loader, epoch)
            
            # Update metrics history
            self.metrics_history['epoch'].append(epoch)
            for k, v in {**train_metrics, **dev_metrics}.items():
                self.metrics_history[k].append(v)

            # Log training progress
            logger.info(f"Epoch {epoch} [Train] - Loss: {train_metrics['train_loss']:.3f}")
            logger.info(f"Epoch {epoch} [Dev] - Loss: {dev_metrics['dev_loss']:.3f} | UAS: {dev_metrics['dev_uas'] * 100:.2f}% | LAS: {dev_metrics['dev_las'] * 100:.2f}%")
                
             # 🎉 NEW: Track best UAS and LAS
            if not hasattr(self, 'best_uas'):
                self.best_uas = -float('inf')
            if not hasattr(self, 'best_las'):
                self.best_las = -float('inf')

            if dev_metrics['dev_uas'] > self.best_uas:
                self.best_uas = dev_metrics['dev_uas']
                logger.info("🎉 New best UAS achieved!")

            if dev_metrics['dev_las'] > self.best_las:
                self.best_las = dev_metrics['dev_las']
                logger.info("🎉 New best LAS achieved!")
            
            # Save checkpoint
            self.tracker.save_checkpoint(epoch, self.model, self.optimizer, self.metrics_history)
            
            # Early stopping check
            if self.early_stopper.update(dev_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch} because no improvement in: {self.early_stopper.metrics}")
                break
                
        # Generate final plots
        self.generate_plots()
        return self.metrics_history
        
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in progress_bar:
            # Prepare batch
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            word_map = batch["word_to_subword_map"].to(self.device)
            gold_heads = batch["heads"].to(self.device)
            gold_labels = batch["labels"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            edge_scores, label_scores = self.model(input_ids, attention_mask, word_map)
            
            # Loss calculation
            loss = self.loss_fn(edge_scores, label_scores, gold_heads, gold_labels)
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update progress
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}"
            })
            
        avg_loss = total_loss / len(train_loader)
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        return {'train_loss': avg_loss}
        
    def evaluate(self, eval_loader, epoch, mode='dev'):
        self.model.eval()
        all_pred_heads, all_gold_heads = [], []
        all_pred_labels, all_gold_labels = [], []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Epoch {epoch} [Eval]"):
                # Prepare batch
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                word_map = batch["word_to_subword_map"].to(self.device)
                gold_heads = batch["heads"].to(self.device)
                gold_labels = batch["labels"].to(self.device)
                
                # Forward pass
                edge_scores, label_scores = self.model(input_ids, attention_mask, word_map)
                loss = self.loss_fn(edge_scores, label_scores, gold_heads, gold_labels)
                total_loss += loss.item()
                
                # Get predictions
                pred_heads = torch.argmax(edge_scores, dim=-1)
                batch_indices = torch.arange(pred_heads.size(0))[:, None]
                token_indices = torch.arange(pred_heads.size(1))[None, :]
                gathered_labels = label_scores[batch_indices, token_indices, pred_heads]
                pred_labels = torch.argmax(gathered_labels, dim=-1)
                
                # Filter padding
                valid_mask = (gold_heads != -1)
                all_pred_heads.extend(pred_heads[valid_mask].cpu().tolist())
                all_gold_heads.extend(gold_heads[valid_mask].cpu().tolist())
                all_pred_labels.extend(pred_labels[valid_mask].cpu().tolist())
                all_gold_labels.extend(gold_labels[valid_mask].cpu().tolist())
        
        # Calculate metrics
        avg_loss = total_loss / len(eval_loader)
        uas = self._calculate_uas(all_pred_heads, all_gold_heads)
        las = self._calculate_las(all_pred_heads, all_gold_heads, all_pred_labels, all_gold_labels)
        
        # Log metrics
        self.writer.add_scalar(f'Loss/{mode}', avg_loss, epoch)
        self.writer.add_scalar(f'UAS/{mode}', uas, epoch)
        self.writer.add_scalar(f'LAS/{mode}', las, epoch)

         # Log results
        logger.info(f"Evaluation Results - Loss: {avg_loss:.3f} | UAS: {uas*100:.2f}% | LAS: {las*100:.2f}%")
        
        return {
            f'{mode}_loss': avg_loss,
            f'{mode}_uas': uas,
            f'{mode}_las': las
        }
        
    def generate_plots(self):
        """Generate and save all visualization plots"""
        metrics = self.metrics_history
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(metrics['epoch'], metrics['train_loss'], label='Train')
        plt.plot(metrics['epoch'], metrics['dev_loss'], label='Dev')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        # UAS plot
        plt.subplot(2, 2, 2)
        plt.plot(metrics['epoch'], metrics['dev_uas'])
        plt.title('Unlabeled Attachment Score')
        plt.xlabel('Epoch')
        plt.ylim(0, 1)
        
        # LAS plot
        plt.subplot(2, 2, 3)
        plt.plot(metrics['epoch'], metrics['dev_las'])
        plt.title('Labeled Attachment Score')
        plt.xlabel('Epoch')
        plt.ylim(0, 1)
        
        # Combined UAS/LAS
        plt.subplot(2, 2, 4)
        plt.plot(metrics['epoch'], metrics['dev_uas'], label='UAS')
        plt.plot(metrics['epoch'], metrics['dev_las'], label='LAS')
        plt.title('Attachment Scores')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(self.tracker.plots_dir, 
                               f"{self.tracker.current_run_id}_metrics.png")
        plt.savefig(plot_path)
        plt.close()
        
    def _calculate_uas(self, pred_heads, gold_heads):
        return sum(p == g for p, g in zip(pred_heads, gold_heads)) / len(gold_heads)
        
    def _calculate_las(self, pred_heads, gold_heads, pred_labels, gold_labels):
        correct = sum((p_h == g_h) and (p_l == g_l) 
                  for p_h, g_h, p_l, g_l in zip(pred_heads, gold_heads, pred_labels, gold_labels))
        return correct / len(gold_heads)

def run_hyperparameter_search(config):
    """Run complete hyperparameter search over all loss functions"""
    # Load data once
    train_loader, dev_loader, test_loader, label2id, id2label = get_data_loaders(
        train_file=config['train_file'],
        dev_file=config['dev_file'],
        test_file=config['test_file'],
        label_vocab_file=config['label_vocab'],
        batch_size=config['batch_size']
    )
    
    device = torch.device(config['device'])
    all_results = {}
    
    # Define search space
    hp_configs = { 
        'arc_margin': [
            {'margin': 0.1, 'alpha': 0.5},
            {'margin': 0.3, 'alpha': 0.5},
            {'margin': 0.5, 'alpha': 0.5}
        ],
        'label_smooth': [  # New configs for label smoothing
            {'smoothing': 0.1, 'alpha': 0.5},
            {'smoothing': 0.2, 'alpha': 0.5},
            {'smoothing': 0.3, 'alpha': 0.5}
        ]
    }
    
    for loss_type, configs in hp_configs.items():
        logger.info(f"\n=== Starting {loss_type} experiments ===")
        loss_results = []
        
        for hp_config in configs:
            # Setup experiment tracking
            exp_dir = os.path.join(config['log_dir'], f"{loss_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(exp_dir, exist_ok=True)
            tracker = ExperimentTracker(exp_dir)
            tracker.init_experiment(loss_type, hp_config)
            
            # Initialize model
            model = DependencyParser(
                model_name=config['model_name'],
                num_labels=len(label2id),
                hidden_dim=config['hidden_dim']
            )
            
            # Configure optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=hp_config.get('lr', 1e-5))
            
            # Initialize loss
            if loss_type == 'ce':
                loss_fn = MultiTaskLoss(label2id, loss_type='ce', alpha=hp_config['alpha'])
            elif loss_type == 'focal':
                loss_fn = MultiTaskLoss(label2id, loss_type='focal', gamma=hp_config['gamma'])
            elif loss_type == 'arc_margin':
                loss_fn = ArcMarginLoss(margin=hp_config['margin'], alpha=hp_config['alpha'])
            elif loss_type == 'label_smooth':
                loss_fn = MultiTaskLoss(
                    label2id, 
                    loss_type='label_smooth', 
                    alpha=hp_config['alpha'],
                    smoothing=hp_config['smoothing']
                )
            
            # Check for existing checkpoint
            checkpoint = tracker.load_checkpoint(loss_type, hp_config)
            if checkpoint:
                logger.info(f"Resuming from checkpoint at epoch {checkpoint['epoch']}")
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                initial_epoch = checkpoint['epoch'] + 1
            else:
                initial_epoch = 0
            


            # Create trainer
            trainer = DependencyParserTrainer(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                label2id=label2id,
                id2label=id2label,
                config={**config, 'log_dir': exp_dir}
            )
            trainer.current_epoch = initial_epoch

            # Initialize experiment tracking BEFORE training
            trainer.tracker.init_experiment(loss_type, hp_config)  # <---- Add this line
            
            # Run training
            metrics = trainer.train(train_loader, dev_loader, config['num_epochs'])
            
            # Final evaluation
            test_metrics = trainer.evaluate(test_loader, trainer.current_epoch, mode='test')
            
            # Save results
            result = {
                'hp_config': hp_config,
                'metrics': metrics,
                'test_metrics': test_metrics,
                'best_epoch': np.argmax(metrics['dev_uas'])
            }
            loss_results.append(result)
            
            with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
                json.dump(result, f, cls=NpEncoder, indent=2)
                
        all_results[loss_type] = loss_results
        
        # Generate comparison plots for this loss type
        generate_comparison_plots(loss_results, loss_type, config['log_dir'])
    
    return all_results

def generate_comparison_plots(results, loss_type, output_dir):
    """Generate comparison plots across hyperparameters for one loss type"""
    plt.figure(figsize=(15, 5))
    
    # UAS comparison
    plt.subplot(1, 3, 1)
    for i, res in enumerate(results):
        plt.plot(res['metrics']['dev_uas'], label=f"Config {i+1}")
    plt.title(f'{loss_type} UAS Comparison')
    plt.xlabel('Epoch')
    plt.legend()
    
    # LAS comparison
    plt.subplot(1, 3, 2)
    for i, res in enumerate(results):
        plt.plot(res['metrics']['dev_las'], label=f"Config {i+1}")
    plt.title(f'{loss_type} LAS Comparison')
    plt.xlabel('Epoch')
    
    # Loss comparison
    plt.subplot(1, 3, 3)
    for i, res in enumerate(results):
        plt.plot(res['metrics']['dev_loss'], label=f"Config {i+1}")
    plt.title(f'{loss_type} Loss Comparison')
    plt.xlabel('Epoch')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{loss_type}_comparison.png")
    plt.savefig(plot_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    # Run complete hyperparameter search
    results = run_hyperparameter_search(config)
    
    # Save final results
    with open(os.path.join(config['log_dir'], 'final_results.json'), 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()