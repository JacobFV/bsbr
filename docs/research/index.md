# Research Documentation

This section contains research documents, experimental results, and technical analyses related to the BSBR (Block Sparse with Block Retrieval) architecture.

## Available Research Documents

- [Background on BSBR Architecture](background.md) - Theoretical foundations and design principles of the BSBR architecture
- [Benchmarks](benchmarks.md) - Performance benchmarks of BSBR compared to other attention mechanisms
- [Experiments](experiments.md) - Experimental results from various tests and configurations
- [BSBR Conversion Research](bsbr_conversion_research.md) - Research on converting pre-trained models to BSBR architecture
- [BSBR Conversion Evaluation](bsbr_conversion_evaluation.md) - Comprehensive evaluation of converted BSBR models

## BSBR Conversion Evaluation

Our most recent research has focused on evaluating the performance and behavior of models converted from standard transformers to BSBR architecture. Key findings include:

- **Performance**: At moderate sequence lengths (≤1024 tokens), BSBR doesn't yet show performance advantages on CPU, but theoretical advantages are expected at longer sequences
- **Scaling**: Original transformer scales as O(n^0.34) vs. BSBR as O(n^0.55) in our tests, contrary to expectations
- **Output Similarity**: Significant divergence in output behavior, with negative cosine similarity and 0% agreement in next-token predictions
- **Use Cases**: BSBR conversion is most suitable for very long context processing where approximate outputs are acceptable

[Read the full evaluation →](bsbr_conversion_evaluation.md)

## Overview of Research Focus Areas

1. **Architectural Innovations**
   - Block-sparse attention patterns
   - Efficient retrieval mechanisms
   - Computational complexity improvements

2. **Conversion of Pre-trained Models**
   - Weight transfer methodologies
   - Equivalence preservation
   - Fine-tuning requirements

3. **Performance Analysis**
   - Speed benchmarks
   - Memory efficiency
   - Scaling behavior

4. **Output and Behavior Analysis**
   - Output distribution comparison
   - Attention pattern visualization
   - Next-token prediction agreement 