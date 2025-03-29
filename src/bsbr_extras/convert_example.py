import torch
import argparse
from typing import Optional
from pathlib import Path
import time

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer

from bsbr_extras.converter import convert_to_bsbr


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a GPT model to BSBR and compare performance")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="gpt2", 
        help="HuggingFace model name or path (default: gpt2)"
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=128, 
        help="Chunk size for BSBR (default: 128)"
    )
    parser.add_argument(
        "--compression_factor", 
        type=int, 
        default=None, 
        help="Compression factor for BSBR state vectors (default: None)"
    )
    parser.add_argument(
        "--seq_len", 
        type=int, 
        default=1024, 
        help="Sequence length for comparison (default: 1024)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./converted_model", 
        help="Output directory for the converted model (default: ./converted_model)"
    )
    parser.add_argument(
        "--save_model", 
        action="store_true", 
        help="Save the converted model"
    )
    
    return parser.parse_args()


def test_performance(model, tokenizer, seq_len: int, device: str = "cuda"):
    """Test the model's performance on a random sequence."""
    # Generate a random input sequence
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device=device)
    attention_mask = torch.ones(1, seq_len, device=device)
    
    # Measure processing time
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    end_time = time.time()
    
    return end_time - start_time


def main():
    args = parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Use a smaller model by default to avoid memory issues
    model_name = args.model_name if args.model_name != "gpt2-medium" else "gpt2"
    print(f"Loading model: {model_name}")
    
    # Load original model and tokenizer
    original_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Convert model to BSBR
    print(f"Converting model to BSBR (chunk_size={args.chunk_size}, compression_factor={args.compression_factor})")
    try:
        bsbr_base_model = convert_to_bsbr(
            model_name,
            chunk_size=args.chunk_size,
            compression_factor=args.compression_factor
        ).to(device)
        
        # Create a simple wrapper to make BSBR model compatible with the original interface
        class BSBRModelWrapper(torch.nn.Module):
            def __init__(self, bsbr_model, original_lm_head):
                super().__init__()
                self.bsbr_model = bsbr_model
                self.lm_head = original_lm_head
                
            def forward(self, input_ids, attention_mask=None, **kwargs):
                hidden_states = self.bsbr_model(input_ids, attention_mask)
                lm_logits = self.lm_head(hidden_states)
                return lm_logits
        
        # Create wrapper model
        bsbr_model = BSBRModelWrapper(bsbr_base_model, original_model.lm_head).to(device)
        
        # Compare parameters
        orig_params = sum(p.numel() for p in original_model.parameters())
        bsbr_params = sum(p.numel() for p in bsbr_model.parameters())
        print(f"Original model parameters: {orig_params:,}")
        print(f"BSBR model parameters: {bsbr_params:,}")
        print(f"Parameter difference: {bsbr_params - orig_params:,} ({(bsbr_params / orig_params - 1) * 100:.2f}%)")
        
        # Test performance
        print(f"\nTesting performance on sequence length: {args.seq_len}")
        
        # Use a smaller sequence length for warm-up to avoid memory issues
        warmup_seq_len = min(args.seq_len // 2, 256)
        
        # Warm up
        print("Warming up...")
        for _ in range(3):
            test_performance(original_model, tokenizer, warmup_seq_len, device)
            test_performance(bsbr_model, tokenizer, warmup_seq_len, device)
        
        # Original model
        print("Testing original model...")
        time_original = test_performance(original_model, tokenizer, args.seq_len, device)
        print(f"Original model processing time: {time_original:.4f} seconds")
        
        # BSBR model
        print("Testing BSBR model...")
        time_bsbr = test_performance(bsbr_model, tokenizer, args.seq_len, device)
        print(f"BSBR model processing time: {time_bsbr:.4f} seconds")
        
        # Calculate speedup
        speedup = time_original / time_bsbr
        print(f"Speedup: {speedup:.2f}x")
        
        # Test longer sequence (if possible)
        longer_seq_len = min(args.seq_len * 2, 2048)  # Limit to 2048 to avoid memory issues
        try:
            print(f"\nTesting performance on longer sequence length: {longer_seq_len}")
            
            # Original model (may OOM)
            try:
                print("Testing original model...")
                time_original_long = test_performance(original_model, tokenizer, longer_seq_len, device)
                print(f"Original model processing time: {time_original_long:.4f} seconds")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("Original model ran out of memory for the longer sequence.")
                    time_original_long = None
                else:
                    raise
            
            # BSBR model
            print("Testing BSBR model...")
            time_bsbr_long = test_performance(bsbr_model, tokenizer, longer_seq_len, device)
            print(f"BSBR model processing time: {time_bsbr_long:.4f} seconds")
            
            if time_original_long is not None:
                speedup_long = time_original_long / time_bsbr_long
                print(f"Speedup on longer sequence: {speedup_long:.2f}x")
        except Exception as e:
            print(f"Error testing longer sequence: {e}")
        
        # Save the converted model if requested
        if args.save_model:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the model
            torch.save(bsbr_model.state_dict(), output_dir / "bsbr_model.pt")
            
            # Save the configuration
            config = {
                "base_model": model_name,
                "chunk_size": args.chunk_size,
                "compression_factor": args.compression_factor,
                "vocab_size": tokenizer.vocab_size,
                "hidden_dim": bsbr_base_model.hidden_dim,
                "num_layers": len(bsbr_base_model.layers),
                "num_heads": bsbr_base_model.layers[0].attention.num_heads
            }
            
            import json
            with open(output_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            print(f"Model and configuration saved to {output_dir}")
            
    except Exception as e:
        print(f"Error converting model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 