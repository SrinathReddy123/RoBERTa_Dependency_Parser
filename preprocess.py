import os
import json
import logging
from conllu import parse_incr
from transformers import RobertaTokenizerFast
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import traceback
from typing import List, Dict, Set, Optional, Any

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ud_preprocessor")
debug_logger = logging.getLogger("ud_debug")
debug_logger.setLevel(logging.DEBUG)

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
MAX_SEQ_LENGTH = 128

def extract_unique_labels(conllu_file_path):
    """Extracts unique dependency labels from a CoNLL-U formatted dataset."""
    logger.info(f"Extracting unique dependency labels from: {conllu_file_path}")
    unique_labels = set()

    try:
        with open(conllu_file_path, "r", encoding="utf-8") as f:
            for sentence in parse_incr(f):
                for token in sentence:
                    if "deprel" in token:
                        unique_labels.add(token["deprel"])

        if not unique_labels:
            logger.error("❌ No dependency labels found! Please check dataset format.")
            raise ValueError("No unique dependency labels extracted. Dataset might be corrupted.")

        logger.info(f"✅ Extracted {len(unique_labels)} unique dependency labels.")
        return unique_labels

    except FileNotFoundError:
        logger.error(f"❌ File not found: {conllu_file_path}")
        raise
    except Exception as e:
        logger.error(f"❌ An error occurred: {str(e)}", exc_info=True)
        raise

class SentenceProcessor:
    def __init__(self, label_set: Set[str]):
        self.label_set = label_set
        self.sample_counter = 0
        self.max_samples_to_log = 10

    def log_sample_data(self, data: Dict):
        if self.sample_counter < self.max_samples_to_log:
            debug_logger.debug("Sample %d:", self.sample_counter + 1)
            debug_logger.debug("Tokens: %s", data['original_tokens'])
            debug_logger.debug("POS tags: %s", data['pos_tags'])
            debug_logger.debug("Original Heads: %s", data['original_heads'])
            debug_logger.debug("Adjusted Heads: %s", data['heads'])
            debug_logger.debug("Labels: %s", data['labels'])
            debug_logger.debug("Input IDs: %s", data['input_ids'])
            debug_logger.debug("Attention Mask: %s", data['attention_mask'])
            debug_logger.debug("Word-Subword Map: %s\n", data['word_to_subword_map'])
            self.sample_counter += 1

    def _pad_sequence(self, sequence: List, pad_value: Any) -> List:
        """Pad sequence to MAX_SEQ_LENGTH"""
        if len(sequence) < MAX_SEQ_LENGTH:
            return sequence + [pad_value] * (MAX_SEQ_LENGTH - len(sequence))
        return sequence[:MAX_SEQ_LENGTH]

def validate_and_fix_tree(heads: List[int], pos_tags: List[str], sentence_id: str) -> bool:
    """Enhanced tree validation with linguistic heuristics"""
    n = len(heads)
    root_indices = [i for i, h in enumerate(heads) if h == -1]
    
    # Fix missing root
    if not root_indices:
        debug_logger.warning(f"{sentence_id}: No root found, attempting correction")
        # Strategy 1: Find first verb
        verb_candidates = [i for i, pos in enumerate(pos_tags) if pos.startswith('V')]
        if verb_candidates:
            new_root = verb_candidates[0]
            heads[new_root] = -1
            debug_logger.info(f"{sentence_id}: Assigned verb '{pos_tags[new_root]}' at position {new_root} as root")
            return True
        
        # Strategy 2: Find first nominal subject
        nominal_candidates = [i for i, pos in enumerate(pos_tags) 
                            if pos in {'NOUN', 'PROPN', 'PRON'}]
        if nominal_candidates:
            new_root = nominal_candidates[0]
            heads[new_root] = -1
            debug_logger.warning(f"{sentence_id}: Assigned nominal '{pos_tags[new_root]}' at {new_root} as root")
            return True
        
        # Fallback: Set first token as root
        heads[0] = -1
        debug_logger.error(f"{sentence_id}: Forced first token as root")
        return True

    # Fix multiple roots
    if len(root_indices) > 1:
        debug_logger.warning(f"{sentence_id}: Multiple roots at {root_indices}, correcting")
        main_root = root_indices[0]
        for i in root_indices[1:]:
            heads[i] = main_root
        return True

    # Validate head indices
    for i, h in enumerate(heads):
        if h != -1 and (h < 0 or h >= n):
            debug_logger.warning(f"{sentence_id}: Invalid head {h} for token {i}, clipping")
            heads[i] = max(0, min(h, n-1))
    
    return True

def preprocess_sentence(sentence: str, sentence_id: int, processor: SentenceProcessor) -> Optional[Dict[str, Any]]:
    """Enhanced sentence processing with strict length validation"""
    try:
        lines = [l.strip() for l in sentence.split('\n') if l.strip() and not l.startswith('#')]
        if not lines:
            return None

        tokens, heads, labels, pos_tags = [], [], [], []
        
        for line in lines:
            parts = line.split('\t')
            if len(parts) < 10 or '-' in parts[0] or '.' in parts[0]:
                continue

            try:
                tokens.append(parts[1])
                pos_tags.append(parts[3])
                head = int(parts[6]) if parts[6] != '_' else 0
                heads.append(-1 if head == 0 else head - 1)
                labels.append(parts[7])
            except (ValueError, IndexError) as e:
                debug_logger.warning(f"Sentence {sentence_id}: {str(e)}")
                continue

        if not tokens:
            return None

        # Validate and fix dependency tree
        orig_heads = heads.copy()
        if not validate_and_fix_tree(heads, pos_tags, f"Sent{sentence_id}"):
            return None

        # Tokenize with RoBERTa
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )

        # Create word-subword mapping
        word_to_subword = [-1] * len(tokens)
        for subw_idx, word_idx in enumerate(encoding.word_ids()):
            if word_idx is not None and word_to_subword[word_idx] == -1:
                word_to_subword[word_idx] = subw_idx

        # Handle unmapped words
        for w in range(len(tokens)):
            if word_to_subword[w] == -1:
                word_to_subword[w] = word_to_subword[w-1] if w > 0 else 0

        # Pad all sequences to MAX_SEQ_LENGTH
        result = {
            "sentence_id": sentence_id,
            "input_ids": encoding['input_ids'][0].tolist(),
            "attention_mask": encoding['attention_mask'][0].tolist(),
            "heads": processor._pad_sequence(heads, -1),
            "labels": processor._pad_sequence(labels, "[PAD]"),
            "word_to_subword_map": processor._pad_sequence(word_to_subword, -1),
            "original_heads": processor._pad_sequence(orig_heads, -1),
            "original_tokens": tokens,
            "pos_tags": pos_tags
        }

        # Validate all sequences have correct length
        assert all(len(result[key]) == MAX_SEQ_LENGTH 
                 for key in ['input_ids', 'attention_mask', 'heads', 
                            'labels', 'word_to_subword_map', 'original_heads']), \
               f"Length mismatch in sentence {sentence_id}"

        processor.label_set.update(labels)
        processor.log_sample_data(result)
        
        return result

    except Exception as e:
        logger.error(f"Error processing sentence {sentence_id}: {str(e)}")
        return None

def preprocess_data(conllu_file: str, output_file: str, label_set: Set[str]):
    processor = SentenceProcessor(label_set)
    
    with open(conllu_file, 'r', encoding='utf-8') as f:
        raw_sentences = f.read().strip().split('\n\n')
    
    with Pool(cpu_count()) as pool:
        args = [(sent, idx, processor) for idx, sent in enumerate(raw_sentences)]
        results = list(tqdm(pool.starmap(preprocess_sentence, args), total=len(raw_sentences)))

    processed_data = [r for r in results if r is not None]
    
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)

    return processor.label_set

if __name__ == "__main__":
    # Paths to EWT dataset splits
    train_file = "../data/raw_data/en_ewt-ud-train.conllu"
    dev_file = "../data/raw_data/en_ewt-ud-dev.conllu"
    test_file = "../data/raw_data/en_ewt-ud-test.conllu"

    # Output directories
    output_dir = "processed_data/ewt"
    os.makedirs(output_dir, exist_ok=True)

    # Extract unique labels from ALL splits
    logger.info("Extracting unique labels from training data...")
    label_set = extract_unique_labels(train_file)
    logger.info("Extracting unique labels from dev data...")
    label_set.update(extract_unique_labels(dev_file))
    logger.info("Extracting unique labels from test data...")
    label_set.update(extract_unique_labels(test_file))

    # Preprocess splits
    logger.info("Preprocessing training data...")
    preprocess_data(train_file, os.path.join(output_dir, "train.json"), label_set)
    logger.info("Preprocessing dev data...")
    preprocess_data(dev_file, os.path.join(output_dir, "dev.json"), label_set)
    logger.info("Preprocessing test data...")
    preprocess_data(test_file, os.path.join(output_dir, "test.json"), label_set)

    # Save label vocabulary
    label_vocab_file = os.path.join(output_dir, "label_vocab.json")
    with open(label_vocab_file, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(label_set)), f)
    logger.info(f"Label vocabulary saved to {label_vocab_file}")