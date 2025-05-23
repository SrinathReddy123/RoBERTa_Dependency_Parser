import json
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple

# Configure logging
logger = logging.getLogger("data_utils")
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler('data_processing.log')
console_handler = logging.StreamHandler()

# Set formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# data_utils.py

class DependencyParsingDataset(Dataset):
    def __init__(self, data_file: str, label2id: Dict[str, int], max_length: int = 128):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.label2id = label2id
        self.max_length = max_length
        self.pad_label_id = label2id["[PAD]"]
        
        # Validate all sequence fields have consistent lengths
        for item in self.data:
            self._validate_item_lengths(item, max_length)
            self._validate_labels(item['labels'])

    def __len__(self) -> int:
        return len(self.data)

    def _validate_item_lengths(self, item, max_length):
        """Ensure all sequence fields have exactly max_length"""
        fields_to_check = [
            'input_ids', 'attention_mask', 'heads',
            'labels', 'word_to_subword_map', 'original_heads'
        ]
        
        for field in fields_to_check:
            if len(item[field]) != max_length:
                raise ValueError(
                    f"Field {field} has length {len(item[field])} "
                    f"instead of {max_length} in sample {item.get('sentence_id', 'unknown')}"
                )

    def _validate_labels(self, labels):
        """Ensure all labels exist in vocabulary"""
        for label in labels:
            if label not in self.label2id:
                raise ValueError(f"Label {label} not found in vocabulary")

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'heads': torch.tensor(item['heads'], dtype=torch.long),
            'labels': torch.tensor([self.label2id[label] for label in item['labels']], dtype=torch.long),
            'word_to_subword_map': torch.tensor(item['word_to_subword_map'], dtype=torch.long),
            'original_heads': torch.tensor(item['original_heads'], dtype=torch.long),
            'original_tokens': item['original_tokens'],
            'pos_tags': item['pos_tags']
        }

def collate_fn(batch: List[Dict]) -> Dict:
    """Simplified collate assuming all sequences are pre-padded"""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'heads': torch.stack([item['heads'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'word_to_subword_map': torch.stack([item['word_to_subword_map'] for item in batch]),
        'original_heads': torch.stack([item['original_heads'] for item in batch]),
        'original_tokens': [item['original_tokens'] for item in batch],
        'pos_tags': [item['pos_tags'] for item in batch]
    }

def load_label_vocab(label_vocab_file: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load and validate label vocabulary"""
    with open(label_vocab_file, 'r', encoding='utf-8') as f:
        label_vocab = json.load(f)
    
    if "[PAD]" not in label_vocab:
        label_vocab.append("[PAD]")
    
    label2id = {label: idx for idx, label in enumerate(label_vocab)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    return label2id, id2label


def get_data_loaders(
    train_file: str,
    dev_file: str,
    test_file: str,
    label_vocab_file: str,
    batch_size: int = 32,
    max_length: int = 128
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    
    label2id, id2label = load_label_vocab(label_vocab_file)
    
    # Create datasets
    train_dataset = DependencyParsingDataset(train_file, label2id, max_length)
    dev_dataset = DependencyParsingDataset(dev_file, label2id, max_length)
    test_dataset = DependencyParsingDataset(test_file, label2id, max_length)
    
    # Create DataLoaders with our custom collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Disable multiprocessing for debugging
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    return train_loader, dev_loader, test_loader, label2id, id2label