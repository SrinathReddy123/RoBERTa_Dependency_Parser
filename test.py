import os
import json
import torch
from model import DependencyParser, MultiTaskLoss
from data_utils import get_data_loaders
from trainer import DependencyParserTrainer

def find_best_checkpoint(loss_type, experiment_dir="experiments"):
    """Finds the best checkpoint based on dev UAS performance"""

    candidate_dirs = [d for d in os.listdir(experiment_dir) if d.startswith(loss_type)]
    
    if not candidate_dirs:
        raise FileNotFoundError(f"No experiment directories found for {loss_type}")
    
    best_checkpoint_path = None
    best_uas = -1.0

    # Search through each experiment run
    for exp_subdir in candidate_dirs:
        ckpt_dir = os.path.join(experiment_dir, exp_subdir, "checkpoints")
        
        if not os.path.exists(ckpt_dir):
            print(f"Checkpoint dir missing in {exp_subdir}")
            continue
        
        all_checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]

        for ckpt_file in all_checkpoints:
            ckpt_path = os.path.join(ckpt_dir, ckpt_file)

            try:
                data = torch.load(ckpt_path, map_location='cpu')
                dev_uas = data['metrics']['dev_uas'][-1]
                
                if dev_uas > best_uas:
                    best_uas = dev_uas
                    best_checkpoint_path = ckpt_path

            except Exception as e:
                print(f"Failed to load {ckpt_path}: {e}")

    if best_checkpoint_path is None:
        raise FileNotFoundError(f"No valid checkpoints found for {loss_type}")
    
    print(f"Found best checkpoint: {best_checkpoint_path} with Dev UAS: {best_uas:.4f}")
    return best_checkpoint_path


def evaluate_on_test_set(config, model_path, loss_type):
    """Evaluate a saved model on test set"""

    # Load data
    _, _, test_loader, label2id, id2label = get_data_loaders(
        train_file=config['train_file'],
        dev_file=config['dev_file'],
        test_file=config['test_file'],
        label_vocab_file=config['label_vocab'],
        batch_size=config['batch_size']
    )

    # Load model
    model = DependencyParser(
        model_name=config['model_name'],
        num_labels=len(label2id),
        hidden_dim=config['hidden_dim']
    ).to(config['device'])

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=config['device'])
    model.load_state_dict(checkpoint['model_state'])

    # Reconstruct loss_fn according to loss_type used in training
    if loss_type == 'ce':
        loss_fn = MultiTaskLoss(label2id, loss_type='ce', alpha=0.5)
    elif loss_type == 'focal':
        loss_fn = MultiTaskLoss(label2id, loss_type='focal', gamma=2.0, alpha=0.5)
    elif loss_type == 'arc_margin':
        loss_fn = MultiTaskLoss(label2id, loss_type='arc_margin', margin=0.3, alpha=0.5)
    elif loss_type == 'label_smooth':
        loss_fn = MultiTaskLoss(label2id, loss_type='label_smooth', smoothing=0.1, alpha=0.5)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Initialize trainer
    trainer = DependencyParserTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=None,
        device=config['device'],
        label2id=label2id,
        id2label=id2label,
        config=config
    )

    # Evaluate
    test_metrics = trainer.evaluate(test_loader, 0, mode='test')

    # Print results
    print(f"\nTest Performance ({loss_type}):")
    print(f"UAS: {test_metrics['test_uas'] * 100:.2f}%")
    print(f"LAS: {test_metrics['test_las'] * 100:.2f}%")
    print(f"Loss: {test_metrics['test_loss']:.3f}")

    # Save results
    result_path = f"test_results_{loss_type}.json"
    with open(result_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"✅ Results saved to {result_path}")
    return test_metrics


if __name__ == "__main__":
    # Load config
    with open("config.json") as f:
        config = json.load(f)

    # Test all loss types
    for loss_type in ['arc_margin', 'label_smooth']:
        try:
            print(f"\n{'='*50}")
            print(f"Evaluating {loss_type} model...")

            # Find best checkpoint for this loss type
            best_model_path = find_best_checkpoint(loss_type)

            # Run evaluation
            evaluate_on_test_set(config, best_model_path, loss_type)

        except Exception as e:
            print(f"❌ Error evaluating {loss_type}: {str(e)}")

# 'ce', 'focal', 'arc_margin', 'label_smooth'

