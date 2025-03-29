# Advanced Usage Examples

This guide demonstrates advanced usage patterns and configurations for the BSBR model.

## Custom Attention Configuration

```python
from bsbr import BSBRModel, BSBRAttention

# Create custom attention layer
custom_attention = BSBRAttention(
    hidden_dim=512,
    num_heads=8,
    chunk_size=128,
    dropout=0.1,
    compression_factor=4  # Enable state compression
)

# Create model with custom attention
model = BSBRModel(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=4,
    num_heads=8,
    chunk_size=128,
    ff_dim=2048,
    dropout=0.1,
    attention_layer=custom_attention  # Use custom attention
)
```

## Memory-Efficient Training

```python
import torch
from bsbr import BSBRModel
from torch.cuda.amp import autocast, GradScaler

# Create model
model = BSBRModel(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=4,
    num_heads=8,
    chunk_size=128,
    ff_dim=2048,
    dropout=0.1
)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Use mixed precision training
scaler = GradScaler()

def train_with_mixed_precision(model, dataloader, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            with autocast():
                outputs = model(batch['input_ids'], batch['attention_mask'])
                loss = criterion(outputs, batch['labels'])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

## Custom Chunking Strategy

```python
from bsbr import BSBRModel
from bsbr.utils.chunking import CustomChunkingStrategy

# Create custom chunking strategy
chunking_strategy = CustomChunkingStrategy(
    chunk_size=128,
    overlap=32,  # Overlap between chunks
    stride=96    # Stride for sliding window
)

# Create model with custom chunking
model = BSBRModel(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=4,
    num_heads=8,
    chunk_size=128,
    ff_dim=2048,
    dropout=0.1,
    chunking_strategy=chunking_strategy
)
```

## Multi-GPU Training

```python
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from bsbr import BSBRModel

def setup_ddp():
    # Initialize distributed training
    torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    return local_rank

def train_ddp():
    local_rank = setup_ddp()
    
    # Create model
    model = BSBRModel(
        vocab_size=10000,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        chunk_size=128,
        ff_dim=2048,
        dropout=0.1
    )
    
    # Wrap model in DDP
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # Create distributed dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=4
    )
    
    # Training loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            # Training step
            ...
```

## Custom Model Variants

```python
from bsbr import BSBRModel
from bsbr_extras import LinearTransformer, DeltaNet

class HybridModel(BSBRModel):
    """Hybrid model combining BSBR and Linear attention."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_layer = LinearTransformer(
            vocab_size=kwargs['vocab_size'],
            hidden_dim=kwargs['hidden_dim'],
            num_layers=1,
            num_heads=kwargs['num_heads'],
            ff_dim=kwargs['ff_dim'],
            dropout=kwargs['dropout']
        )
    
    def forward(self, input_ids, attention_mask=None):
        # BSBR processing
        bsbr_output = super().forward(input_ids, attention_mask)
        
        # Linear attention processing
        linear_output = self.linear_layer(input_ids, attention_mask)
        
        # Combine outputs
        return (bsbr_output + linear_output) / 2

# Create and use hybrid model
model = HybridModel(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=4,
    num_heads=8,
    chunk_size=128,
    ff_dim=2048,
    dropout=0.1
)
```

## Performance Optimization

```python
from bsbr import BSBRModel
from bsbr.utils.optimization import optimize_for_inference

# Create model
model = BSBRModel(
    vocab_size=10000,
    hidden_dim=512,
    num_layers=4,
    num_heads=8,
    chunk_size=128,
    ff_dim=2048,
    dropout=0.1
)

# Optimize model for inference
optimized_model = optimize_for_inference(model)

# Use torch.jit for further optimization
scripted_model = torch.jit.script(optimized_model)

# Benchmark performance
def benchmark_model(model, input_ids, num_runs=100):
    model.eval()
    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = model(input_ids)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(input_ids)
        torch.cuda.synchronize()
        end_time = time.time()
        
        return (end_time - start_time) / num_runs
``` 