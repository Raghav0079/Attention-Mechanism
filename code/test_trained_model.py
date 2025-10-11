"""
Test script for evaluating trained Transformer model on WMT 14 dataset
Optimized for CPU usage
"""

import os
import torch
import argparse
import time
from tqdm import tqdm
from sacrebleu import BLEU
import random
import numpy as np

from transformer import Transformer
from wmt14_dataset import WMT14Dataset, create_wmt14_dataloaders


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def create_masks(src, tgt, pad_idx):
    """Create source and target masks"""
    # Source mask
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_len)
    
    # Target mask
    tgt_len = tgt.size(1)
    tgt_padding_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_len)
    look_ahead_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
    look_ahead_mask = look_ahead_mask.to(tgt.device)
    
    tgt_mask = tgt_padding_mask & ~look_ahead_mask
    return src_mask, tgt_mask


def greedy_translate(model, src_ids, src_mask, dataset, max_length=100, device='cpu'):
    """
    Translate using greedy search (much more memory efficient than beam search)
    """
    model.eval()
    
    # Constants
    batch_size = src_ids.size(0)
    pad_idx = dataset.src_tokenizer.token_to_id(dataset.pad_token)
    bos_idx = dataset.tgt_tokenizer.token_to_id(dataset.bos_token)
    eos_idx = dataset.tgt_tokenizer.token_to_id(dataset.eos_token)
    
    # Encode source
    encoder_output = model.encoder(src_ids, src_mask)
    
    # Start with <bos> token for each batch
    tgt_ids = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
    
    # Generate translations
    all_translations = []
    
    for i in range(max_length):
        # Create target mask
        tgt_len = tgt_ids.size(1)
        look_ahead_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(device)
        tgt_padding_mask = (tgt_ids != pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_padding_mask & ~look_ahead_mask
        
        # Decode
        decoder_output = model.decoder(tgt_ids, encoder_output, src_mask, tgt_mask)
        
        # Get next token (greedy)
        next_token = decoder_output[:, -1].argmax(dim=-1, keepdim=True)
        tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
        
        # Check if all sequences have generated EOS token
        if ((next_token == eos_idx).sum() == batch_size) or (i == max_length - 1):
            break
    
    # Decode translations
    translations = []
    for i in range(batch_size):
        tokens = tgt_ids[i].cpu().tolist()
        translation = dataset.decode_sequence(tokens, is_source=False)
        translation = translation.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
        translations.append(translation)
    
    return translations


def beam_search_translate(model, src_ids, src_mask, dataset, beam_size=3, max_length=100, device='cpu'):
    """Translate using beam search for better quality (only use with very small batches)"""
    model.eval()
    
    # Constants
    batch_size = src_ids.size(0)
    pad_idx = dataset.src_tokenizer.token_to_id(dataset.pad_token)
    bos_idx = dataset.tgt_tokenizer.token_to_id(dataset.bos_token)
    eos_idx = dataset.tgt_tokenizer.token_to_id(dataset.eos_token)
    
    # Encode source
    encoder_output = model.encoder(src_ids, src_mask)
    
    # Start with <bos> token for each batch
    start_tokens = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
    
    # Beam search for each sentence in batch
    all_translations = []
    
    for batch_idx in range(batch_size):
        # Get encoder output for this sentence
        enc_output = encoder_output[batch_idx:batch_idx+1]  # keep batch dimension
        src_mask_i = src_mask[batch_idx:batch_idx+1]
        
        # Initialize beam with <bos>
        beam = [(start_tokens[batch_idx:batch_idx+1], 0.0, False)]  # (tokens, score, completed)
        
        # Beam search iterations (much more memory intensive than greedy search)
        for _ in range(max_length):
            next_beam = []
            
            # Process each sequence in current beam
            for tokens, score, completed in beam:
                if completed:
                    # Keep completed sequences
                    next_beam.append((tokens, score, completed))
                    continue
                
                # Create target mask
                tgt_len = tokens.size(1)
                look_ahead_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(device)
                tgt_padding_mask = (tokens != pad_idx).unsqueeze(1).unsqueeze(2)
                tgt_mask = tgt_padding_mask & ~look_ahead_mask
                
                # Decode current sequence
                decoder_output = model.decoder(tokens, enc_output, src_mask_i, tgt_mask)
                logits = decoder_output[:, -1]  # last token predictions
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Get top-k next tokens
                topk_probs, topk_indices = log_probs.topk(beam_size)
                
                # Add to beam
                for i in range(beam_size):
                    next_token = topk_indices[0, i:i+1].unsqueeze(0)
                    next_score = score + topk_probs[0, i].item()
                    next_tokens = torch.cat([tokens, next_token], dim=1)
                    
                    # Check if sequence is completed
                    is_completed = (next_token.item() == eos_idx)
                    next_beam.append((next_tokens, next_score, is_completed))
            
            # Sort and keep top-k beams
            next_beam = sorted(next_beam, key=lambda x: x[1], reverse=True)[:beam_size]
            beam = next_beam
            
            # Check if all beams are completed
            if all(completed for _, _, completed in beam):
                break
        
        # Get best translation (highest score)
        best_tokens = beam[0][0].squeeze().cpu().tolist()
        translation = dataset.decode_sequence(best_tokens, is_source=False)
        translation = translation.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
        all_translations.append(translation)
    
    return all_translations


def evaluate_model(model, test_loader, dataset, num_samples=None, device='cpu'):
    """
    Evaluate the model on the test set with minimal memory usage
    """
    print(f"\nEvaluating model on test set...")
    model.eval()
    model.to(device)
    
    pad_idx = dataset.src_tokenizer.token_to_id(dataset.pad_token)
    
    all_references = []
    all_translations = []
    
    start_time = time.time()
    
    # Only process subset of test data if specified
    sample_count = 0
    batch_count = 0
    
    with torch.no_grad():
        # Use tqdm for progress tracking
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Process only a small number of batches to avoid memory issues
            if num_samples and batch_count >= (num_samples // test_loader.batch_size + 1):
                break
                
            batch_count += 1
            
            # Move batch to device one sample at a time to save memory
            batch_size = batch['src'].size(0)
            for i in range(batch_size):
                # Skip if we've collected enough samples
                if num_samples and len(all_translations) >= num_samples:
                    break
                    
                # Extract single sample
                src = batch['src'][i:i+1].to(device)
                tgt = batch['tgt'][i:i+1].to(device)
                
                # Get reference translation
                reference = dataset.decode_sequence(tgt[0], is_source=False)
                reference = reference.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
                
                # Source mask
                src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
                
                # Use greedy search instead of beam search to save memory
                translation = greedy_translate(
                    model, src, src_mask, dataset,
                    max_length=100,
                    device=device
                )[0]
                
                all_translations.append(translation)
                all_references.append(reference)
                
                # Show a few examples
                if sample_count < 3:
                    src_text = dataset.decode_sequence(src[0], is_source=True)
                    src_text = src_text.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
                    
                    print(f"\nExample {sample_count + 1}:")
                    print(f"Source:     {src_text}")
                    print(f"Reference:  {reference}")
                    print(f"Prediction: {translation}")
                
                sample_count += 1
                
                # Clean up to save memory
                del src, tgt, src_mask
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Clean up batch data to save memory
            del batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # Calculate BLEU score
    bleu = BLEU()
    bleu_score = bleu.corpus_score(all_translations, [all_references]).score
    
    eval_time = time.time() - start_time
    
    print(f"\nEvaluation Results:")
    print(f"BLEU Score: {bleu_score:.2f}")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    print(f"Samples evaluated: {len(all_translations)}")
    
    return bleu_score, all_translations, all_references


def translate_sentences(model, dataset, sentences, device='cpu'):
    """
    Translate a list of English sentences to German
    """
    model.eval()
    model.to(device)
    
    pad_idx = dataset.src_tokenizer.token_to_id(dataset.pad_token)
    
    # Encode source sentences
    encoded_sentences = [dataset.src_tokenizer.encode(sent) for sent in sentences]
    src_ids = [torch.tensor(enc.ids, device=device) for enc in encoded_sentences]
    
    # Pad source sentences
    max_len = max(len(ids) for ids in src_ids)
    padded_src = torch.ones(len(sentences), max_len, dtype=torch.long, device=device) * pad_idx
    for i, ids in enumerate(src_ids):
        padded_src[i, :len(ids)] = ids
    
    # Create source masks
    src_mask = (padded_src != pad_idx).unsqueeze(1).unsqueeze(2)
    
    # Translate
    translations = beam_search_translate(
        model, padded_src, src_mask, dataset,
        beam_size=3,  # Small beam size for CPU efficiency
        max_length=100,
        device=device
    )
    
    return translations


def load_best_model(model, checkpoint_dir, device='cpu'):
    """
    Load the best model based on validation BLEU score
    """
    # Find checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('best_model_epoch') and f.endswith('.pth')]
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Extract BLEU scores from filenames
    checkpoint_scores = []
    for file in checkpoint_files:
        try:
            bleu_score = float(file.split('_bleu_')[1].split('.pth')[0])
            checkpoint_scores.append((file, bleu_score))
        except:
            continue
    
    # Find best checkpoint
    if not checkpoint_scores:
        # Fall back to the last checkpoint if no BLEU scores available
        checkpoint_files.sort()
        best_checkpoint = checkpoint_files[-1]
    else:
        # Get checkpoint with highest BLEU score
        best_checkpoint = sorted(checkpoint_scores, key=lambda x: x[1], reverse=True)[0][0]
    
    # Load checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint)
    print(f"Loading best model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint_path


def main():
    parser = argparse.ArgumentParser(description='Test trained Transformer model on WMT 14 (memory optimized)')
    parser.add_argument('--checkpoint_dir', type=str, default='./wmt14_checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--num_test_samples', type=int, default=10,  # Reduced default
                        help='Number of test samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=1,  # Single sample batch size
                        help='Batch size for evaluation')
    parser.add_argument('--max_length', type=int, default=64,
                        help='Maximum sequence length')
    parser.add_argument('--vocab_size', type=int, default=16000,  # Reduced vocabulary
                        help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=4,  # Reduced heads
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,  # Reduced layers
                        help='Number of encoder/decoder layers')
    parser.add_argument('--d_ff', type=int, default=512,  # Reduced feed-forward dim
                        help='Feed-forward network dimension')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation, skip interactive mode')
    parser.add_argument('--interactive_only', action='store_true',
                        help='Only run interactive mode, skip evaluation')
    args = parser.parse_args()
    
    # Set reproducibility
    set_seed(42)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset with minimal memory footprint
    print("Loading WMT 14 test dataset...")
    try:
        dataset = WMT14Dataset(
            split='test',
            max_length=args.max_length,
            vocab_size=args.vocab_size,
            cache_dir='./wmt14_data',  # Use the same cache dir as during training
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying with default cache directory...")
        dataset = WMT14Dataset(
            split='test',
            max_length=args.max_length,
            vocab_size=args.vocab_size,
        )
    
    # Create test loader with minimal settings
    if not args.interactive_only:
        print("Creating test dataloader...")
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    # Create model with reduced architecture for memory efficiency
    src_vocab_size, tgt_vocab_size = dataset.get_vocab_sizes()
    print(f"Creating model with src_vocab={src_vocab_size}, tgt_vocab={tgt_vocab_size}")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_length=args.max_length,
        dropout=0.1
    )
    
    # Load best model
    print("Loading model checkpoint...")
    try:
        model, checkpoint_path = load_best_model(model, args.checkpoint_dir, device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Please ensure you have trained the model and have checkpoints available.")
        return
        
    # Only run evaluation if not in interactive-only mode
    if not args.interactive_only:
        # Evaluate on test set with minimal samples
        print(f"Evaluating on {args.num_test_samples} test samples...")
        try:
            bleu_score, translations, references = evaluate_model(
                model, test_loader, dataset, 
                num_samples=args.num_test_samples, 
                device=device
            )
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            print("Skipping evaluation and proceeding to interactive mode...")
    
    # Interactive testing
    print("\n" + "="*50)
    print("Interactive Translation Demo")
    print("="*50)
    print("Type English sentences to translate or 'q' to quit")
    
    while True:
        user_input = input("\nEnglish: ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
        
        translations = translate_sentences(model, dataset, [user_input], device)
        print(f"German: {translations[0]}")


if __name__ == "__main__":
    main()