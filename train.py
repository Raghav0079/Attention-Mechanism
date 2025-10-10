import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from transformer import Transformer
import math


class DummyTranslationDataset(Dataset):
    """
    A simple dummy dataset for sequence-to-sequence translation
    This simulates a translation task where input sequences are mapped to output sequences
    """
    def __init__(self, num_samples=1000, src_vocab_size=1000, tgt_vocab_size=1000, 
                 max_length=20, pad_idx=0):
        self.num_samples = num_samples
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_length = max_length
        self.pad_idx = pad_idx
        
        # Generate dummy data
        self.data = []
        for _ in range(num_samples):
            src_len = np.random.randint(5, max_length)
            tgt_len = np.random.randint(5, max_length)
            
            # Create source sequence (avoid pad_idx)
            src = torch.randint(1, src_vocab_size, (src_len,))
            src_padded = torch.full((max_length,), pad_idx, dtype=torch.long)
            src_padded[:src_len] = src
            
            # Create target sequence (avoid pad_idx) 
            tgt = torch.randint(1, tgt_vocab_size, (tgt_len,))
            tgt_padded = torch.full((max_length,), pad_idx, dtype=torch.long)
            tgt_padded[:tgt_len] = tgt
            
            self.data.append((src_padded, tgt_padded))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss as used in the original Transformer paper
    """
    def __init__(self, vocab_size, smoothing=0.1, pad_idx=0):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.pad_idx = pad_idx
        
    def forward(self, pred, target):
        # pred: (batch_size * seq_len, vocab_size)
        # target: (batch_size * seq_len,)
        
        batch_size, seq_len, vocab_size = pred.shape
        pred = pred.view(-1, vocab_size)
        target = target.view(-1)
        
        # Create smoothed labels
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (vocab_size - 2))  # -2 for true label and pad
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        true_dist[:, self.pad_idx] = 0
        
        # Mask padded positions
        mask = (target != self.pad_idx)
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        true_dist = true_dist[mask]
        pred = pred[mask]
        
        return self.criterion(torch.log_softmax(pred, dim=-1), true_dist)


def create_masks(src, tgt, pad_idx=0):
    """Create masks for source and target sequences"""
    batch_size, src_len = src.shape
    _, tgt_len = tgt.shape
    
    # Source padding mask
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_len)
    
    # Target padding mask
    tgt_padding_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_len)
    
    # Target look-ahead mask
    look_ahead_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
    look_ahead_mask = look_ahead_mask.to(tgt.device)
    
    # Combine target masks
    tgt_mask = tgt_padding_mask & ~look_ahead_mask
    
    return src_mask, tgt_mask


class NoamScheduler:
    """
    Learning rate scheduler used in the original Transformer paper
    """
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(self.step_num ** (-0.5), 
                                          self.step_num * self.warmup_steps ** (-1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def train_model(model, train_loader, num_epochs=10, device='cpu'):
    """
    Train the Transformer model
    """
    model.to(device)
    model.train()
    
    # Loss function with label smoothing
    criterion = LabelSmoothingLoss(vocab_size=model.decoder.output_projection.out_features, 
                                  smoothing=0.1, pad_idx=0)
    
    # Optimizer (Adam with custom learning rate schedule)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=model.encoder.d_model, warmup_steps=4000)
    
    losses = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            # Prepare input and target
            tgt_input = tgt[:, :-1]  # Remove last token for input
            tgt_output = tgt[:, 1:]  # Remove first token for target
            
            # Create masks
            src_mask, tgt_mask = create_masks(src, tgt_input)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(src, tgt_input, src_mask, tgt_mask)
            
            # Calculate loss
            loss = criterion(output, tgt_output)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            lr = scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {lr:.6f}')
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}')
    
    return losses


def plot_training_loss(losses):
    """Plot training loss over epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Transformer Training Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def inference_example(model, src_vocab_size, tgt_vocab_size, max_length=20, device='cpu'):
    """
    Simple inference example showing how to use the trained model
    """
    model.eval()
    
    # Create a dummy source sequence
    src = torch.randint(1, src_vocab_size, (1, 10)).to(device)  # batch_size=1, seq_len=10
    
    # Start with start-of-sequence token (assume it's token 1)
    tgt = torch.tensor([[1]]).to(device)  # Start token
    
    print("Inference example:")
    print(f"Source sequence: {src.squeeze().tolist()}")
    
    with torch.no_grad():
        for i in range(max_length - 1):
            # Create masks
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
            tgt_len = tgt.size(1)
            look_ahead_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(device)
            tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
            tgt_mask = tgt_padding_mask & ~look_ahead_mask
            
            # Forward pass
            output = model(src, tgt, src_mask, tgt_mask)
            
            # Get next token prediction
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Append to target sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if end-of-sequence token (assume it's token 2)
            if next_token.item() == 2:
                break
    
    print(f"Generated sequence: {tgt.squeeze().tolist()}")


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model hyperparameters
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1
    
    # Create model
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, 
                       num_layers, d_ff, max_seq_length, dropout)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    # Create dataset and dataloader
    dataset = DummyTranslationDataset(num_samples=1000, src_vocab_size=src_vocab_size, 
                                     tgt_vocab_size=tgt_vocab_size, max_length=20)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train model
    losses = train_model(model, dataloader, num_epochs=5, device=device)
    
    # Plot training loss
    plot_training_loss(losses)
    
    # Run inference example
    inference_example(model, src_vocab_size, tgt_vocab_size, device=device)
    
    # Save model
    torch.save(model.state_dict(), 'transformer_model.pth')
    print("Model saved as 'transformer_model.pth'")