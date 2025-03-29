# Research Examples

This guide demonstrates how to use BSBR for research and experimentation.

## Model Comparison

```python
from bsbr.evals import compare_models, analyze_results
import matplotlib.pyplot as plt

# Compare different models across various sequence lengths
results = compare_models(
    seq_lengths=[64, 128, 256, 512, 1024],
    models=['BSBR', 'Linear', 'DeltaNet', 'SlidingWindow', 'Standard'],
    metrics=['inference_time', 'memory_usage', 'accuracy']
)

# Analyze results
analysis = analyze_results(results)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(results['seq_lengths'], results['BSBR']['inference_time'], label='BSBR')
plt.plot(results['seq_lengths'], results['Standard']['inference_time'], label='Standard')
plt.xlabel('Sequence Length')
plt.ylabel('Inference Time (s)')
plt.title('Inference Time vs Sequence Length')
plt.legend()
plt.show()
```

## Memory Analysis

```python
from bsbr.utils.memory import analyze_memory_usage
import torch

def profile_memory_usage(model, input_ids):
    """Profile memory usage during forward pass."""
    torch.cuda.reset_peak_memory_stats()
    
    # Forward pass
    outputs = model(input_ids)
    
    # Get memory stats
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    current_memory = torch.cuda.memory_allocated() / 1024**2   # MB
    
    return {
        'peak_memory': peak_memory,
        'current_memory': current_memory
    }

# Analyze memory usage across different chunk sizes
chunk_sizes = [32, 64, 128, 256]
memory_results = {}

for chunk_size in chunk_sizes:
    model = BSBRModel(
        vocab_size=10000,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        chunk_size=chunk_size,
        ff_dim=2048,
        dropout=0.1
    )
    
    input_ids = torch.randint(0, 10000, (1, 1024))
    memory_results[chunk_size] = profile_memory_usage(model, input_ids)
```

## Attention Visualization

```python
from bsbr.utils.visualization import visualize_attention

def analyze_attention_patterns(model, input_ids):
    """Analyze attention patterns in the model."""
    # Get attention weights
    attention_weights = model.get_attention_weights(input_ids)
    
    # Visualize attention patterns
    plt.figure(figsize=(12, 8))
    visualize_attention(attention_weights)
    plt.title('Attention Patterns')
    plt.show()
    
    # Analyze sparsity
    sparsity = (attention_weights == 0).float().mean()
    print(f"Attention sparsity: {sparsity:.2%}")

# Compare attention patterns across models
models = {
    'BSBR': BSBRModel(...),
    'Linear': LinearTransformer(...),
    'Standard': StandardTransformer(...)
}

for name, model in models.items():
    print(f"\nAnalyzing {name} attention patterns:")
    analyze_attention_patterns(model, input_ids)
```

## Scaling Analysis

```python
from bsbr.evals import scaling_analysis

def analyze_scaling_behavior():
    """Analyze how different models scale with sequence length."""
    seq_lengths = [64, 128, 256, 512, 1024, 2048]
    models = ['BSBR', 'Linear', 'Standard']
    
    results = scaling_analysis(
        seq_lengths=seq_lengths,
        models=models,
        metrics=['time', 'memory', 'flops']
    )
    
    # Plot scaling curves
    plt.figure(figsize=(12, 4))
    
    # Time scaling
    plt.subplot(131)
    for model in models:
        plt.plot(seq_lengths, results[model]['time'], label=model)
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (s)')
    plt.title('Time Scaling')
    plt.legend()
    
    # Memory scaling
    plt.subplot(132)
    for model in models:
        plt.plot(seq_lengths, results[model]['memory'], label=model)
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory (GB)')
    plt.title('Memory Scaling')
    plt.legend()
    
    # FLOPs scaling
    plt.subplot(133)
    for model in models:
        plt.plot(seq_lengths, results[model]['flops'], label=model)
    plt.xlabel('Sequence Length')
    plt.ylabel('FLOPs')
    plt.title('FLOPs Scaling')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
```

## Custom Research Experiments

```python
from bsbr.utils.research import ExperimentRunner

class CustomExperiment(ExperimentRunner):
    """Custom research experiment."""
    
    def setup(self):
        """Setup experiment parameters."""
        self.models = {
            'BSBR': BSBRModel(...),
            'Linear': LinearTransformer(...),
            'Standard': StandardTransformer(...)
        }
        self.seq_lengths = [64, 128, 256, 512, 1024]
        self.metrics = ['time', 'memory', 'accuracy']
    
    def run_experiment(self, model, seq_length):
        """Run single experiment."""
        # Generate input
        input_ids = torch.randint(0, 10000, (1, seq_length))
        
        # Measure metrics
        start_time = time.time()
        outputs = model(input_ids)
        inference_time = time.time() - start_time
        
        memory_usage = torch.cuda.max_memory_allocated() / 1024**2
        
        # Calculate accuracy (example)
        accuracy = self.calculate_accuracy(outputs)
        
        return {
            'time': inference_time,
            'memory': memory_usage,
            'accuracy': accuracy
        }
    
    def analyze_results(self, results):
        """Analyze experiment results."""
        # Custom analysis code
        pass

# Run experiment
experiment = CustomExperiment()
results = experiment.run()
experiment.analyze_results(results)
``` 