import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os
from tqdm import tqdm


class WMT14Dataset(Dataset):
    """
    WMT 14 English-German dataset implementation for Transformer training
    """
    
    def __init__(self, split='train', max_length=512, vocab_size=32000, 
                 cache_dir='./data', force_retrain=False):
        """
        Args:
            split: 'train', 'validation', or 'test'
            max_length: Maximum sequence length
            vocab_size: Vocabulary size for BPE tokenization
            cache_dir: Directory to cache tokenizers and processed data
            force_retrain: Force retraining of tokenizers
        """
        self.split = split
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.cache_dir = cache_dir
        
        # Special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load dataset
        print(f"Loading WMT 14 En-De dataset ({split} split)...")
        self.dataset = load_dataset("wmt14", "de-en", split=split, cache_dir=cache_dir)
        
        # Initialize tokenizers
        self.src_tokenizer_path = os.path.join(cache_dir, "src_tokenizer.json")
        self.tgt_tokenizer_path = os.path.join(cache_dir, "tgt_tokenizer.json")
        
        if force_retrain or not os.path.exists(self.src_tokenizer_path) or not os.path.exists(self.tgt_tokenizer_path):
            print("Training BPE tokenizers...")
            self.train_tokenizers()
        else:
            print("Loading existing tokenizers...")
            self.src_tokenizer = Tokenizer.from_file(self.src_tokenizer_path)
            self.tgt_tokenizer = Tokenizer.from_file(self.tgt_tokenizer_path)
        
        # Preprocess dataset
        self.processed_data = self.preprocess_dataset()
        
        print(f"Dataset loaded: {len(self.processed_data)} samples")
    
    def train_tokenizers(self):
        """Train BPE tokenizers for source and target languages"""
        
        # Prepare training data
        src_texts = []
        tgt_texts = []
        
        print("Extracting texts for tokenizer training...")
        for example in tqdm(self.dataset):
            src_texts.append(example['translation']['en'])
            tgt_texts.append(example['translation']['de'])
        
        # Train source tokenizer (English)
        print("Training English tokenizer...")
        self.src_tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        self.src_tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            show_progress=True
        )
        self.src_tokenizer.train_from_iterator(src_texts, trainer)
        
        # Add post-processor to add BOS/EOS tokens
        self.src_tokenizer.post_processor = TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            special_tokens=[(self.bos_token, self.src_tokenizer.token_to_id(self.bos_token)),
                           (self.eos_token, self.src_tokenizer.token_to_id(self.eos_token))]
        )
        
        # Train target tokenizer (German)
        print("Training German tokenizer...")
        self.tgt_tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        self.tgt_tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            show_progress=True
        )
        self.tgt_tokenizer.train_from_iterator(tgt_texts, trainer)
        
        # Add post-processor to add BOS/EOS tokens
        self.tgt_tokenizer.post_processor = TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            special_tokens=[(self.bos_token, self.tgt_tokenizer.token_to_id(self.bos_token)),
                           (self.eos_token, self.tgt_tokenizer.token_to_id(self.eos_token))]
        )
        
        # Save tokenizers
        self.src_tokenizer.save(self.src_tokenizer_path)
        self.tgt_tokenizer.save(self.tgt_tokenizer_path)
        
        print(f"Tokenizers saved to {self.cache_dir}")
    
    def preprocess_dataset(self):
        """Preprocess the dataset by tokenizing and padding sequences"""
        processed_data = []
        
        print("Preprocessing dataset...")
        for example in tqdm(self.dataset):
            src_text = example['translation']['en']
            tgt_text = example['translation']['de']
            
            # Tokenize
            src_encoded = self.src_tokenizer.encode(src_text)
            tgt_encoded = self.tgt_tokenizer.encode(tgt_text)
            
            src_ids = src_encoded.ids
            tgt_ids = tgt_encoded.ids
            
            # Skip sequences that are too long
            if len(src_ids) > self.max_length or len(tgt_ids) > self.max_length:
                continue
            
            # Pad sequences
            src_padded = self.pad_sequence(src_ids, self.max_length)
            tgt_padded = self.pad_sequence(tgt_ids, self.max_length)
            
            processed_data.append({
                'src': torch.tensor(src_padded, dtype=torch.long),
                'tgt': torch.tensor(tgt_padded, dtype=torch.long),
                'src_text': src_text,
                'tgt_text': tgt_text
            })
        
        return processed_data
    
    def pad_sequence(self, sequence, max_length):
        """Pad sequence to max_length with pad_token_id"""
        pad_id = self.src_tokenizer.token_to_id(self.pad_token)
        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            return sequence + [pad_id] * (max_length - len(sequence))
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]
    
    def get_vocab_sizes(self):
        """Get vocabulary sizes for source and target languages"""
        return self.src_tokenizer.get_vocab_size(), self.tgt_tokenizer.get_vocab_size()
    
    def decode_sequence(self, sequence, is_source=True):
        """Decode a tokenized sequence back to text"""
        tokenizer = self.src_tokenizer if is_source else self.tgt_tokenizer
        
        # Remove padding tokens
        sequence = sequence.tolist() if isinstance(sequence, torch.Tensor) else sequence
        pad_id = tokenizer.token_to_id(self.pad_token)
        sequence = [token_id for token_id in sequence if token_id != pad_id]
        
        return tokenizer.decode(sequence)


def create_wmt14_dataloaders(batch_size=32, max_length=512, vocab_size=32000, 
                            cache_dir='./data', num_workers=4, max_train_samples=None):
    """
    Create DataLoaders for WMT 14 train, validation, and test sets
    
    Args:
        batch_size: Batch size for training
        max_length: Maximum sequence length
        vocab_size: Vocabulary size for BPE
        cache_dir: Cache directory for data and tokenizers
        num_workers: Number of workers for data loading
        max_train_samples: Maximum number of training samples to use (None for all)
    
    Returns:
        train_loader, val_loader, test_loader, dataset (for vocab info)
    """
    
    # Create datasets
    train_split = f'train[:{max_train_samples}]' if max_train_samples else 'train'
    train_dataset = WMT14Dataset(train_split, max_length, vocab_size, cache_dir)
    val_dataset = WMT14Dataset('validation', max_length, vocab_size, cache_dir)
    test_dataset = WMT14Dataset('test', max_length, vocab_size, cache_dir)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset


def collate_fn(batch):
    """
    Custom collate function for batching WMT 14 data
    """
    src_batch = torch.stack([item['src'] for item in batch])
    tgt_batch = torch.stack([item['tgt'] for item in batch])
    
    return {
        'src': src_batch,
        'tgt': tgt_batch,
        'src_texts': [item['src_text'] for item in batch],
        'tgt_texts': [item['tgt_text'] for item in batch]
    }


if __name__ == "__main__":
    # Test the dataset implementation
    print("Testing WMT 14 dataset implementation...")
    
    # Create a small dataset for testing
    dataset = WMT14Dataset('train', max_length=128, vocab_size=8000, cache_dir='./test_data')
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocab sizes: {dataset.get_vocab_sizes()}")
    
    # Test a few samples
    for i in range(3):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"Source: {sample['src_text']}")
        print(f"Target: {sample['tgt_text']}")
        print(f"Source tokens: {sample['src'][:20]}...")  # First 20 tokens
        print(f"Target tokens: {sample['tgt'][:20]}...")  # First 20 tokens
        
        # Test decoding
        decoded_src = dataset.decode_sequence(sample['src'], is_source=True)
        decoded_tgt = dataset.decode_sequence(sample['tgt'], is_source=False)
        print(f"Decoded source: {decoded_src}")
        print(f"Decoded target: {decoded_tgt}")
    
    # Test dataloader
    print("\nTesting DataLoader...")
    train_loader, val_loader, test_loader, _ = create_wmt14_dataloaders(
        batch_size=4, 
        max_length=128, 
        vocab_size=8000,
        cache_dir='./test_data',
        num_workers=0  # Set to 0 for testing
    )
    
    batch = next(iter(train_loader))
    print(f"Batch source shape: {batch['src'].shape}")
    print(f"Batch target shape: {batch['tgt'].shape}")
    print(f"Sample source text: {batch['src_texts'][0]}")
    print(f"Sample target text: {batch['tgt_texts'][0]}")