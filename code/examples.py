"""
Examples and Demonstrations of Transformer Components

This script provides interactive examples of the key components
of the "Attention Is All You Need" Transformer architecture.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformer import (
    MultiHeadAttention, 
    PositionalEncoding, 
    Transformer,
    EncoderLayer,
    DecoderLayer
)


def demonstrate_attention_mechanism():
    """
    Demonstrate how the attention mechanism works with a simple example
    """
    print("=" * 60)
    print("ATTENTION MECHANISM DEMONSTRATION")
    print("=" * 60)
    
    # Create a simple sequence
    d_model = 64
    seq_len = 5
    batch_size = 1
    
    # Create multi-head attention layer
    mha = MultiHeadAttention(d_model, num_heads=4)
    
    # Create input sequence (representing word embeddings)
    # Let's imagine this represents: "The cat sat on mat"
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input sequence shape: {x.shape}")
    print("Sequence represents: ['The', 'cat', 'sat', 'on', 'mat']")
    
    # Apply attention
    output, attention_weights = mha(x, x, x)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Visualize attention weights for the first head
    attention_matrix = attention_weights[0, 0].detach().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_matrix, cmap='Blues')
    plt.colorbar()
    plt.title('Attention Weights (Head 1)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    words = ['The', 'cat', 'sat', 'on', 'mat']
    plt.xticks(range(len(words)), words)
    plt.yticks(range(len(words)), words)
    plt.tight_layout()
    plt.show()
    
    # Print attention scores
    print("\nAttention Matrix (Head 1):")
    print("Rows=Query, Cols=Key")
    for i, query_word in enumerate(words):
        print(f"{query_word:4s}: ", end="")
        for j, key_word in enumerate(words):
            print(f"{attention_matrix[i,j]:.3f} ", end="")
        print()


def demonstrate_positional_encoding():
    """
    Demonstrate positional encoding patterns
    """
    print("\n" + "=" * 60)
    print("POSITIONAL ENCODING DEMONSTRATION")
    print("=" * 60)
    
    d_model = 64
    max_len = 50
    
    pe = PositionalEncoding(d_model, max_len)
    
    # Create dummy input
    x = torch.zeros(max_len, 1, d_model)
    pos_encoded = pe(x)
    
    # Extract positional encodings
    pos_encodings = pos_encoded[:, 0, :].detach().numpy()
    
    # Visualize positional encodings
    plt.figure(figsize=(12, 8))
    
    # Plot the positional encoding matrix
    plt.subplot(2, 2, 1)
    plt.imshow(pos_encodings.T, cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.title('Positional Encoding Matrix')
    plt.xlabel('Position')
    plt.ylabel('Encoding Dimension')
    
    # Plot specific dimensions over positions
    plt.subplot(2, 2, 2)
    dims_to_plot = [0, 1, 4, 8, 16, 32]
    for dim in dims_to_plot:
        plt.plot(pos_encodings[:, dim], label=f'dim {dim}')
    plt.title('Positional Encoding - Selected Dimensions')
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.legend()
    
    # Show sine and cosine patterns for early dimensions
    plt.subplot(2, 2, 3)
    plt.plot(pos_encodings[:, 0], label='sin(pos/10000^(0/64))', linewidth=2)
    plt.plot(pos_encodings[:, 1], label='cos(pos/10000^(0/64))', linewidth=2)
    plt.plot(pos_encodings[:, 2], label='sin(pos/10000^(2/64))', linewidth=2)
    plt.plot(pos_encodings[:, 3], label='cos(pos/10000^(2/64))', linewidth=2)
    plt.title('Sine/Cosine Patterns (First Few Dimensions)')
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.legend()
    
    # Show frequency analysis
    plt.subplot(2, 2, 4)
    frequencies = []
    for i in range(0, d_model, 2):
        freq = 1 / (10000 ** (i / d_model))
        frequencies.append(freq)
    
    plt.semilogy(range(0, d_model, 2), frequencies, 'o-')
    plt.title('Encoding Frequencies')
    plt.xlabel('Dimension')
    plt.ylabel('Frequency (log scale)')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Positional encoding shape: {pos_encodings.shape}")
    print("Each position gets a unique encoding based on sine/cosine functions")
    print("Lower dimensions have higher frequencies, higher dimensions have lower frequencies")


def demonstrate_masking():
    """
    Demonstrate different types of masking used in Transformers
    """
    print("\n" + "=" * 60)
    print("MASKING DEMONSTRATION")
    print("=" * 60)
    
    # Create example sequences with padding
    vocab_size = 100
    pad_idx = 0
    
    # Example batch with different length sequences
    src = torch.tensor([
        [1, 15, 23, 45, 67, 0, 0, 0],  # Length 5, padded with 0s
        [2, 34, 56, 78, 89, 12, 33, 0] # Length 7, padded with 0s
    ])
    
    tgt = torch.tensor([
        [1, 11, 22, 33, 44, 0],  # Length 5, padded with 0s
        [2, 55, 66, 77, 88, 99]  # Length 6, no padding
    ])
    
    print("Source sequences (with padding):")
    for i, seq in enumerate(src):
        non_pad = seq[seq != pad_idx].tolist()
        print(f"  Sequence {i+1}: {non_pad} (original length: {len(non_pad)})")
    
    print("\nTarget sequences (with padding):")
    for i, seq in enumerate(tgt):
        non_pad = seq[seq != pad_idx].tolist()
        print(f"  Sequence {i+1}: {non_pad} (original length: {len(non_pad)})")
    
    # Create Transformer model to use masking functions
    model = Transformer(vocab_size, vocab_size, d_model=64, num_heads=4, num_layers=2)
    
    # 1. Padding masks for source
    src_mask = model.create_padding_mask(src, pad_idx)
    print(f"\nSource padding mask shape: {src_mask.shape}")
    print("Source padding mask (1=attend, 0=ignore):")
    for i in range(src.size(0)):
        print(f"  Sequence {i+1}: {src_mask[i, 0, 0].int().tolist()}")
    
    # 2. Look-ahead mask for target (causal mask)
    tgt_len = tgt.size(1)
    look_ahead_mask = model.create_look_ahead_mask(tgt_len)
    print(f"\nLook-ahead mask shape: {look_ahead_mask.shape}")
    print("Look-ahead mask (prevents attending to future tokens):")
    print(look_ahead_mask.int().numpy())
    
    # Visualize masks
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Source padding mask for first sequence
    axes[0].imshow(src_mask[0, 0, 0].unsqueeze(0).float(), cmap='Blues')
    axes[0].set_title('Source Padding Mask\n(Sequence 1)')
    axes[0].set_xlabel('Source Position')
    axes[0].set_ylabel('Query Position')
    
    # Look-ahead mask
    axes[1].imshow(look_ahead_mask.float(), cmap='Blues')
    axes[1].set_title('Look-ahead Mask\n(Causal Attention)')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    
    # Combined mask example (padding + look-ahead for target)
    tgt_padding_mask = model.create_padding_mask(tgt, pad_idx)
    combined_mask = tgt_padding_mask[0, 0] & look_ahead_mask
    axes[2].imshow(combined_mask.float(), cmap='Blues')
    axes[2].set_title('Combined Mask\n(Padding + Look-ahead)')
    axes[2].set_xlabel('Key Position')
    axes[2].set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.show()


def demonstrate_layer_by_layer():
    """
    Demonstrate what happens at each layer of the Transformer
    """
    print("\n" + "=" * 60)
    print("LAYER-BY-LAYER PROCESSING DEMONSTRATION")
    print("=" * 60)
    
    # Create a simple model
    d_model = 128
    vocab_size = 1000
    
    # Create individual components
    encoder_layer = EncoderLayer(d_model, num_heads=8, d_ff=512)
    decoder_layer = DecoderLayer(d_model, num_heads=8, d_ff=512)
    
    # Create sample inputs
    batch_size, seq_len = 2, 8
    src_tokens = torch.randint(1, vocab_size, (batch_size, seq_len))
    tgt_tokens = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    # Create embeddings (simplified)
    embedding = torch.nn.Embedding(vocab_size, d_model)
    pos_encoding = PositionalEncoding(d_model)
    
    # Process through encoder
    print("ENCODER PROCESSING:")
    print("-" * 30)
    
    # Input embeddings + positional encoding
    src_embedded = embedding(src_tokens) * np.sqrt(d_model)
    src_embedded = pos_encoding(src_embedded.transpose(0, 1)).transpose(0, 1)
    print(f"1. Input embeddings + positional encoding: {src_embedded.shape}")
    
    # Pass through encoder layer
    encoder_output = encoder_layer(src_embedded)
    print(f"2. After encoder layer: {encoder_output.shape}")
    
    # Process through decoder
    print("\nDECODER PROCESSING:")
    print("-" * 30)
    
    # Target embeddings + positional encoding  
    tgt_embedded = embedding(tgt_tokens) * np.sqrt(d_model)
    tgt_embedded = pos_encoding(tgt_embedded.transpose(0, 1)).transpose(0, 1)
    print(f"1. Target embeddings + positional encoding: {tgt_embedded.shape}")
    
    # Create causal mask for decoder
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    
    # Pass through decoder layer
    decoder_output = decoder_layer(tgt_embedded, encoder_output, tgt_mask=causal_mask)
    print(f"2. After decoder layer: {decoder_output.shape}")
    
    # Final linear projection (to vocabulary)
    output_projection = torch.nn.Linear(d_model, vocab_size)
    logits = output_projection(decoder_output)
    print(f"3. After output projection: {logits.shape}")
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    print(f"4. After softmax (probabilities): {probs.shape}")
    
    print(f"\nFinal output represents probability distribution over {vocab_size} vocabulary tokens")
    print(f"for each of the {seq_len} positions in each of the {batch_size} sequences")


def demonstrate_complete_example():
    """
    Complete end-to-end example with a tiny vocabulary
    """
    print("\n" + "=" * 60)
    print("COMPLETE END-TO-END EXAMPLE")
    print("=" * 60)
    
    # Create tiny vocabulary for demonstration
    vocab = {
        '<pad>': 0, '<start>': 1, '<end>': 2,
        'hello': 3, 'world': 4, 'how': 5, 'are': 6, 'you': 7
    }
    inv_vocab = {v: k for k, v in vocab.items()}
    
    vocab_size = len(vocab)
    
    # Create model
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        dropout=0.1
    )
    
    # Create example sequences
    # Source: "hello world"
    # Target: "how are you"
    src_seq = torch.tensor([[vocab['hello'], vocab['world'], vocab['<pad>'], vocab['<pad>']]])
    tgt_seq = torch.tensor([[vocab['<start>'], vocab['how'], vocab['are'], vocab['you']]])
    
    print("Example Translation Task:")
    print(f"Source: {[inv_vocab[idx.item()] for idx in src_seq[0] if idx.item() != 0]}")
    print(f"Target: {[inv_vocab[idx.item()] for idx in tgt_seq[0]]}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        # Create masks
        src_mask = model.create_padding_mask(src_seq)
        tgt_mask = model.create_look_ahead_mask(tgt_seq.size(1))
        
        # Forward pass
        output = model(src_seq, tgt_seq, src_mask, tgt_mask)
        
        print(f"\nModel output shape: {output.shape}")
        
        # Get predictions
        predictions = output.argmax(dim=-1)
        
        print("Model predictions:")
        for pos in range(tgt_seq.size(1)):
            actual_token = inv_vocab[tgt_seq[0, pos].item()]
            predicted_token = inv_vocab[predictions[0, pos].item()]
            prob = F.softmax(output[0, pos], dim=0)[predictions[0, pos]].item()
            print(f"  Position {pos}: actual='{actual_token}', predicted='{predicted_token}' (prob={prob:.3f})")


if __name__ == "__main__":
    print("Transformer Architecture Demonstrations")
    print("=====================================")
    
    # Run all demonstrations
    try:
        demonstrate_attention_mechanism()
        demonstrate_positional_encoding()  
        demonstrate_masking()
        demonstrate_layer_by_layer()
        demonstrate_complete_example()
        
        print("\n" + "=" * 60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Some demonstrations may require matplotlib for visualization.")
        print("Install with: pip install matplotlib")