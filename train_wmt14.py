import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from transformer import Transformer
from wmt14_dataset import create_wmt14_dataloaders, WMT14Dataset
import math
import os
from tqdm import tqdm
import time
from sacrebleu import BLEU


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
        batch_size, seq_len, vocab_size = pred.shape
        # Use reshape instead of view to handle non-contiguous tensors
        pred = pred.reshape(-1, vocab_size)
        target = target.reshape(-1)
        
        # Create smoothed labels
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        true_dist[:, self.pad_idx] = 0
        
        # Mask padded positions
        mask = (target != self.pad_idx)
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        
        true_dist = true_dist[mask]
        pred = pred[mask]
        
        return self.criterion(torch.log_softmax(pred, dim=-1), true_dist)


def create_masks(src, tgt, pad_idx=0):
    """Create masks for source and target sequences"""
    batch_size, src_len = src.shape
    _, tgt_len = tgt.shape
    
    # Source padding mask
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    
    # Target padding mask
    tgt_padding_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    
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


def calculate_bleu(predictions, references, dataset):
    """Calculate BLEU score for translations"""
    bleu = BLEU()
    
    decoded_preds = []
    decoded_refs = []
    
    for pred, ref in zip(predictions, references):
        # Decode predictions (remove special tokens)
        pred_text = dataset.decode_sequence(pred, is_source=False)
        ref_text = dataset.decode_sequence(ref, is_source=False)
        
        # Remove special tokens for BLEU calculation
        pred_text = pred_text.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
        ref_text = ref_text.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
        
        decoded_preds.append(pred_text)
        decoded_refs.append(ref_text)
    
    # Calculate BLEU score
    score = bleu.corpus_score(decoded_preds, [decoded_refs])
    return score.score


def train_model(model, train_loader, val_loader, dataset, num_epochs=10, device='cpu', 
                save_dir='./checkpoints', accumulation_steps=1, patience=2):
    """
    Train the Transformer model on WMT 14 dataset
    
    Args:
        accumulation_steps: Number of steps to accumulate gradients before updating weights
                            Effectively increases batch size without using more memory
        patience: Number of epochs with no improvement after which training will be stopped
    """
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    
    # Get vocabulary sizes
    src_vocab_size, tgt_vocab_size = dataset.get_vocab_sizes()
    pad_idx = dataset.src_tokenizer.token_to_id(dataset.pad_token)
    
    # Loss function with label smoothing
    criterion = LabelSmoothingLoss(vocab_size=tgt_vocab_size, smoothing=0.1, pad_idx=pad_idx)
    
    # Optimizer (Adam with custom learning rate schedule)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=model.encoder.d_model, warmup_steps=4000)
    
    train_losses = []
    val_losses = []
    val_bleu_scores = []
    
    best_bleu = 0.0
    lr = 0.0  # Initialize learning rate variable
    no_improve_epochs = 0  # For early stopping
    
    print(f"Starting training on WMT 14 En-De dataset...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Source vocab size: {src_vocab_size}")
    print(f"Target vocab size: {tgt_vocab_size}")
    print(f"Gradient accumulation steps: {accumulation_steps} (effective batch size: {train_loader.batch_size * accumulation_steps})")
    
    # Clean up GPU memory if possible
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        # Reset gradients at the start of each epoch
        optimizer.zero_grad()
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, batch in enumerate(train_pbar):
            # Move batch to device 
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            # Prepare input and target
            tgt_input = tgt[:, :-1]  # Remove last token for input
            tgt_output = tgt[:, 1:]  # Remove first token for target
            
            # Create masks
            src_mask, tgt_mask = create_masks(src, tgt_input, pad_idx)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)
            
            # Forward pass
            output = model(src, tgt_input, src_mask, tgt_mask)
            
            # Calculate loss
            loss = criterion(output, tgt_output)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Only update weights after accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                lr = scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{lr:.2e}',
                'Avg Loss': f'{epoch_loss/num_batches:.4f}'
            })
            
            # Log every 1000 steps
            if batch_idx % 1000 == 0 and batch_idx > 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Step {batch_idx}, Loss: {loss.item():.4f}, LR: {lr:.6f}')
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        all_predictions = []
        all_references = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for batch in val_pbar:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                src_mask, tgt_mask = create_masks(src, tgt_input, pad_idx)
                src_mask = src_mask.to(device)
                tgt_mask = tgt_mask.to(device)
                
                output = model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(output, tgt_output)
                
                val_loss += loss.item()
                val_batches += 1
                
                # Collect predictions for BLEU score
                predictions = output.argmax(dim=-1)
                all_predictions.extend(predictions.cpu())
                all_references.extend(tgt_output.cpu())
                
                val_pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Calculate BLEU score
        try:
            bleu_score = calculate_bleu(all_predictions[:100], all_references[:100], dataset)  # Sample for speed
            val_bleu_scores.append(bleu_score)
        except Exception as e:
            print(f"Error calculating BLEU: {e}")
            bleu_score = 0.0
            val_bleu_scores.append(0.0)
        
        print(f'Epoch {epoch+1}/{num_epochs} completed.')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, BLEU: {bleu_score:.2f}')
        
        # Save checkpoint if best BLEU
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            checkpoint_path = os.path.join(save_dir, f'best_model_epoch_{epoch+1}_bleu_{bleu_score:.2f}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'bleu_score': bleu_score,
                'src_vocab_size': src_vocab_size,
                'tgt_vocab_size': tgt_vocab_size
            }, checkpoint_path)
            print(f'New best model saved: {checkpoint_path}')
            no_improve_epochs = 0  # Reset counter when we have a new best
        else:
            no_improve_epochs += 1
            print(f'No improvement for {no_improve_epochs} epochs')
            
            # Early stopping check
            if patience > 0 and no_improve_epochs >= patience:
                print(f'Early stopping after {epoch+1} epochs (no improvement for {patience} epochs)')
                break
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'bleu_score': bleu_score
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    return train_losses, val_losses, val_bleu_scores


def plot_training_curves(train_losses, val_losses, val_bleu_scores, save_path='training_curves.png'):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label='Training Loss', marker='o')
    ax1.plot(epochs, val_losses, label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # BLEU score curve
    ax2.plot(epochs, val_bleu_scores, label='Validation BLEU', marker='^', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('BLEU Score')
    ax2.set_title('Validation BLEU Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def translate_sentences(model, dataset, sentences, device='cpu', max_length=100):
    """
    Translate a list of English sentences to German
    """
    model.eval()
    model.to(device)
    
    translations = []
    pad_idx = dataset.src_tokenizer.token_to_id(dataset.pad_token)
    bos_idx = dataset.tgt_tokenizer.token_to_id(dataset.bos_token)
    eos_idx = dataset.tgt_tokenizer.token_to_id(dataset.eos_token)
    
    with torch.no_grad():
        for sentence in sentences:
            # Encode source sentence
            src_encoded = dataset.src_tokenizer.encode(sentence)
            src_ids = torch.tensor([src_encoded.ids], device=device)
            
            # Create source mask
            src_mask = (src_ids != pad_idx).unsqueeze(1).unsqueeze(2)
            
            # Encode source
            encoder_output = model.encoder(src_ids, src_mask)
            
            # Generate translation
            tgt_ids = torch.tensor([[bos_idx]], device=device)
            
            for _ in range(max_length):
                # Create target mask
                tgt_len = tgt_ids.size(1)
                look_ahead_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(device)
                tgt_padding_mask = (tgt_ids != pad_idx).unsqueeze(1).unsqueeze(2)
                tgt_mask = tgt_padding_mask & ~look_ahead_mask
                
                # Decode
                decoder_output = model.decoder(tgt_ids, encoder_output, src_mask, tgt_mask)
                
                # Get next token
                next_token = decoder_output[:, -1, :].argmax(dim=-1, keepdim=True)
                tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
                
                # Stop if EOS token
                if next_token.item() == eos_idx:
                    break
            
            # Decode translation
            translation = dataset.decode_sequence(tgt_ids.squeeze().cpu(), is_source=False)
            translation = translation.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
            translations.append(translation)
    
    return translations


if __name__ == "__main__":
    # Configuration
    config = {
        'batch_size': 8,          # Reduced batch size
        'max_length': 64,         # Further reduced sequence length
        'vocab_size': 8000,       # Further reduced vocabulary size
        'd_model': 128,           # Further reduced model dimension
        'num_heads': 4,           # Reduced number of heads
        'num_layers': 3,          # Further reduced number of layers
        'd_ff': 512,              # Further reduced feed-forward dimension
        'dropout': 0.1,
        'num_epochs': 5,          # Reduced number of epochs
        'cache_dir': './wmt14_data',
        'save_dir': './wmt14_checkpoints',
        'accumulation_steps': 4,  # Gradient accumulation steps
        'max_train_samples': 10000, # Drastically reduced training samples
        'patience': 2             # Early stopping patience (epochs)
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading WMT 14 dataset...")
    print(f"Using reduced dataset size: max {config.get('max_train_samples', 'all')} samples")
    train_loader, val_loader, test_loader, dataset = create_wmt14_dataloaders(
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        vocab_size=config['vocab_size'],
        cache_dir=config['cache_dir'],
        num_workers=0,  # Set to 0 to reduce memory usage
        max_train_samples=config.get('max_train_samples', None)  # Limit dataset size
    )
    
    # Get vocabulary sizes
    src_vocab_size, tgt_vocab_size = dataset.get_vocab_sizes()
    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {tgt_vocab_size}")
    
    # Create model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_length=config['max_length'],
        dropout=config['dropout']
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    # Train model
    train_losses, val_losses, val_bleu_scores = train_model(
        model, train_loader, val_loader, dataset,
        num_epochs=config['num_epochs'],
        device=device,
        save_dir=config['save_dir'],
        accumulation_steps=config.get('accumulation_steps', 1),  # Enable gradient accumulation
        patience=config.get('patience', 2)  # Enable early stopping
    )
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, val_bleu_scores)
    
    # Test translation
    test_sentences = [
        "Hello, how are you?",
        "The weather is beautiful today.",
        "I love machine learning and artificial intelligence.",
        "This is a test of the neural machine translation system."
    ]
    
    print("\nTesting translations:")
    translations = translate_sentences(model, dataset, test_sentences, device)
    
    for src, tgt in zip(test_sentences, translations):
        print(f"EN: {src}")
        print(f"DE: {tgt}")
        print("-" * 50)