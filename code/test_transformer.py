import torch
import numpy as np
from transformer import Transformer, MultiHeadAttention, PositionalEncoding
import unittest


class TestTransformer(unittest.TestCase):
    """Test cases for Transformer implementation"""
    
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 10
        self.d_model = 512
        self.num_heads = 8
        self.vocab_size = 1000
        
    def test_multi_head_attention(self):
        """Test multi-head attention module"""
        mha = MultiHeadAttention(self.d_model, self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        output, attention_weights = mha(x, x, x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        # Check attention weights shape
        self.assertEqual(attention_weights.shape, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))
        
        # Check attention weights sum to 1 (approximately)
        attention_sum = attention_weights.sum(dim=-1)
        expected_sum = torch.ones_like(attention_sum)
        self.assertTrue(torch.allclose(attention_sum, expected_sum, atol=1e-5))
        
    def test_positional_encoding(self):
        """Test positional encoding"""
        pe = PositionalEncoding(self.d_model, max_seq_length=100)
        x = torch.randn(self.seq_len, self.batch_size, self.d_model)
        
        output = pe(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.seq_len, self.batch_size, self.d_model))
        
        # Check that positional encoding is deterministic
        output2 = pe(x)
        self.assertTrue(torch.allclose(output, output2))
        
    def test_transformer_forward(self):
        """Test complete Transformer forward pass"""
        model = Transformer(
            src_vocab_size=self.vocab_size,
            tgt_vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=2,  # Smaller for testing
            d_ff=1024,
            max_seq_length=100,
            dropout=0.1
        )
        
        src = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        tgt = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
        
        output = model(src, tgt)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
        self.assertEqual(output.shape, expected_shape)
        
    def test_mask_creation(self):
        """Test mask creation functions"""
        model = Transformer(self.vocab_size, self.vocab_size)
        
        # Test padding mask
        seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        padding_mask = model.create_padding_mask(seq, pad_idx=0)
        
        expected_mask = torch.tensor([
            [[[True, True, True, False, False]]],
            [[[True, True, False, False, False]]]
        ])
        self.assertTrue(torch.equal(padding_mask, expected_mask))
        
        # Test look-ahead mask
        size = 5
        look_ahead_mask = model.create_look_ahead_mask(size)
        
        # Check that it's lower triangular
        expected_mask = torch.tril(torch.ones(size, size)) == 1
        self.assertTrue(torch.equal(look_ahead_mask, expected_mask))
        
    def test_attention_with_mask(self):
        """Test attention mechanism with masking"""
        mha = MultiHeadAttention(self.d_model, self.num_heads)
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Create a simple mask (mask out last half of sequence)
        mask = torch.ones(self.batch_size, 1, self.seq_len, self.seq_len)
        mask[:, :, :, self.seq_len//2:] = 0
        
        output, attention_weights = mha(x, x, x, mask)
        
        # Check that masked positions have near-zero attention
        masked_attention = attention_weights[:, :, :, self.seq_len//2:]
        self.assertTrue(torch.all(masked_attention < 1e-6))
        
    def test_parameter_count(self):
        """Test that parameter count is reasonable"""
        model = Transformer(
            src_vocab_size=10000,
            tgt_vocab_size=10000,
            d_model=512,
            num_heads=8,
            num_layers=6,
            d_ff=2048
        )
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Should be around 65M parameters (original paper size)
        self.assertGreater(param_count, 50_000_000)
        self.assertLess(param_count, 100_000_000)
        
        print(f"Total parameters: {param_count:,}")


def test_training_step():
    """Test a single training step"""
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=256,  # Smaller for testing
        num_heads=8,
        num_layers=2,
        d_ff=512
    )
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create dummy batch
    batch_size = 4
    seq_len = 10
    src = torch.randint(1, 1000, (batch_size, seq_len))
    tgt = torch.randint(1, 1000, (batch_size, seq_len))
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    
    output = model(src, tgt_input)
    loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
    
    loss.backward()
    optimizer.step()
    
    print(f"Training step completed. Loss: {loss.item():.4f}")
    
    return loss.item()


def run_performance_test():
    """Test model performance on different input sizes"""
    model = Transformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048
    )
    
    model.eval()
    
    test_cases = [
        (1, 10),    # Single sequence, short
        (1, 50),    # Single sequence, medium
        (8, 20),    # Small batch
        (32, 10),   # Larger batch, short sequences
    ]
    
    print("\nPerformance Test Results:")
    print("Batch Size | Seq Length | Forward Time (ms)")
    print("-" * 45)
    
    with torch.no_grad():
        for batch_size, seq_len in test_cases:
            src = torch.randint(1, 10000, (batch_size, seq_len))
            tgt = torch.randint(1, 10000, (batch_size, seq_len))
            
            # Warm up
            _ = model(src, tgt)
            
            # Time forward pass
            import time
            start_time = time.time()
            for _ in range(10):
                _ = model(src, tgt)
            end_time = time.time()
            
            avg_time_ms = (end_time - start_time) / 10 * 1000
            print(f"{batch_size:10} | {seq_len:10} | {avg_time_ms:12.2f}")


if __name__ == "__main__":
    print("Running Transformer Tests...")
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*50)
    print("Additional Tests")
    print("="*50)
    
    # Test training step
    print("\nTesting training step...")
    test_training_step()
    
    # Run performance test
    print("\nRunning performance tests...")
    run_performance_test()
    
    print("\nAll tests completed successfully!")