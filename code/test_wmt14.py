"""
Quick test script for WMT 14 dataset integration
This script tests dataset loading and basic functionality without full training
"""

import torch
from wmt14_dataset import WMT14Dataset, create_wmt14_dataloaders
from transformer import Transformer


def test_wmt14_small():
    """Test WMT 14 dataset with a small subset"""
    print("Testing WMT 14 dataset integration...")
    
    try:
        # Create a small dataset for testing
        print("1. Creating small test dataset...")
        dataset = WMT14Dataset(
            split='validation',  # Use validation split for faster testing
            max_length=128,
            vocab_size=8000,  # Smaller vocab for faster training
            cache_dir='./test_wmt14_data'
        )
        
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Dataset size: {len(dataset)}")
        print(f"  - Vocab sizes: {dataset.get_vocab_sizes()}")
        
        # Test a few samples
        print("\n2. Testing sample data...")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\nSample {i+1}:")
            print(f"  Source: {sample['src_text'][:100]}...")
            print(f"  Target: {sample['tgt_text'][:100]}...")
            print(f"  Source shape: {sample['src'].shape}")
            print(f"  Target shape: {sample['tgt'].shape}")
        
        # Test dataloader
        print("\n3. Testing DataLoader...")
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(dataloader))
        
        print(f"✓ DataLoader working!")
        print(f"  - Batch source shape: {batch['src'].shape}")
        print(f"  - Batch target shape: {batch['tgt'].shape}")
        
        # Test with small Transformer model
        print("\n4. Testing with small Transformer model...")
        src_vocab_size, tgt_vocab_size = dataset.get_vocab_sizes()
        
        model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=256,  # Smaller model for testing
            num_heads=8,
            num_layers=2,  # Fewer layers for testing
            d_ff=512,
            max_seq_length=128,
            dropout=0.1
        )
        
        # Test forward pass
        src = batch['src'][:2]  # Take first 2 samples
        tgt = batch['tgt'][:2]
        
        # Create simple masks
        pad_idx = dataset.src_tokenizer.token_to_id(dataset.pad_token)
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        
        tgt_input = tgt[:, :-1]
        tgt_len = tgt_input.size(1)
        look_ahead_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
        tgt_padding_mask = (tgt_input != pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_padding_mask & ~look_ahead_mask
        
        with torch.no_grad():
            output = model(src, tgt_input, src_mask, tgt_mask)
            print(f"✓ Model forward pass successful!")
            print(f"  - Output shape: {output.shape}")
        
        print("\n5. Testing tokenization/detokenization...")
        # Test encoding/decoding
        test_sentence = "Hello, this is a test sentence."
        encoded = dataset.src_tokenizer.encode(test_sentence)
        decoded = dataset.decode_sequence(encoded.ids, is_source=True)
        
        print(f"  Original: {test_sentence}")
        print(f"  Encoded:  {encoded.ids[:10]}... (length: {len(encoded.ids)})")
        print(f"  Decoded:  {decoded}")
        
        print(f"\n✅ All tests passed! WMT 14 dataset is ready for training.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_dataset_info():
    """Show information about WMT 14 dataset"""
    print("\n" + "="*60)
    print("WMT 14 English-German Dataset Information")
    print("="*60)
    
    print("""
    The WMT 14 (Workshop on Machine Translation 2014) English-German dataset
    is the same dataset used in the original "Attention Is All You Need" paper.
    
    Dataset Details:
    - Source Language: English
    - Target Language: German  
    - Training samples: ~4.5M sentence pairs
    - Validation samples: ~3K sentence pairs
    - Test samples: ~3K sentence pairs
    
    Preprocessing:
    - BPE (Byte Pair Encoding) tokenization with 32K vocabulary
    - Special tokens: <pad>, <unk>, <s>, </s>
    - Maximum sequence length: 512 tokens (configurable)
    
    This implementation includes:
    ✓ Automatic dataset downloading via Hugging Face datasets
    ✓ BPE tokenizer training and caching
    ✓ Proper padding and masking
    ✓ BLEU score evaluation
    ✓ Checkpointing and model saving
    """)


if __name__ == "__main__":
    show_dataset_info()
    
    # Test the dataset
    success = test_wmt14_small()
    
    if success:
        print(f"\n🎉 Ready to train on WMT 14!")
        print(f"To start full training, run:")
        print(f"  python train_wmt14.py")
        print(f"\nNote: Full training requires significant computational resources.")
        print(f"Consider using a GPU and adjusting batch size accordingly.")
    else:
        print(f"\n❌ Setup incomplete. Please check the error messages above.")
        print(f"You may need to install additional dependencies:")
        print(f"  pip install -r requirements.txt")