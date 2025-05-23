import json
from tqdm import tqdm

def validate_and_fix_data(file_path, max_length=128):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fixed_count = 0
    total_samples = len(data)
    
    for sample in tqdm(data, desc=f"Validating {file_path}"):
        # Fix heads and original_heads
        for field in ['heads', 'original_heads']:
            for i in range(len(sample[field])):
                if sample[field][i] >= max_length:
                    sample[field][i] = max_length - 1
                    fixed_count += 1
        
        # Ensure word_to_subword_map indices are valid
        for i in range(len(sample['word_to_subword_map'])):
            if sample['word_to_subword_map'][i] >= max_length:
                sample['word_to_subword_map'][i] = max_length - 1
                fixed_count += 1
    
    if fixed_count > 0:
        print(f"Fixed {fixed_count} out-of-bounds indices in {total_samples} samples")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    else:
        print(f"All samples validated - no out-of-bounds indices found")

if __name__ == "__main__":
    files_to_validate = [
        "processed_data/ewt/train.json",
        "processed_data/ewt/dev.json",
        "processed_data/ewt/test.json"
    ]
    
    for file_path in files_to_validate:
        validate_and_fix_data(file_path)