# Quantize Strategy

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE-MIT)

Advanced quantization strategies for neural network models, with first-class support for [Candle](https://github.com/huggingface/candle).

- **Research Oriented**

### CLI Usage

```bash
# Basic quantization
quantize_q8k model.safetensors ./output

# With L2 norm permutation
CANDLE_Q8K_PERMUTE=1 quantize_q8k model.safetensors ./output

# With attention-aware strategy
CANDLE_Q8K_PERMUTE=1 CANDLE_Q8K_STRATEGY=attention_aware quantize_q8k model.safetensors ./output
```

### Library Usage

```rust
use quantize_strategy::{QuantizationConfig, StrategyType, quantize_safetensors};

let config = QuantizationConfig {
    strategy_type: StrategyType::AttentionAware,
    use_permutation: true,
    ..Default::default()
};

let result = quantize_safetensors(&input_path, config)?;
println!("Quantized {} tensors with average MSE: {:.2e}", 
         result.tensor_count, result.average_mse);
```

## Validation & Quality

Every quantization includes dual validation:

- **Method 1 (Sum Test)**: Tests overall quantization fidelity
- **Method 2 (Gradient Test)**: Tests permutation-sensitive errors

MSE values typically range from **1e-6 to 1e-4**, indicating excellent quality with minimal accuracy loss.


Typical performance on consumer hardware:
- **~10 tensors/second** with full validation
- **~50% memory reduction** with Q8K quantization  
- **<1% accuracy loss** on standard benchmarks

## Architecture

```
src/
├── core/           # Quantization engine
├── strategies/     # Permutation strategies (L2 norm, etc.)
├── utils/         # Helper functions
└── bin/           # CLI tools
```

### Core Components

- **`QuantizationStrategy`**: Trait for permutation strategies
- **`QuantizationConfig`**: Configuration and parameters
- **`ValidationSystem`**: Dual MSE quality assessment
- **`FileIO`**: Efficient `.q8k` and `.perm` file handling

### Environment Variables

```bash
CANDLE_Q8K_PERMUTE=1          # Enable permutation
CANDLE_Q8K_STRATEGY=l2_norm   # Strategy selection  
CANDLE_Q8K_THREADS=8          # Thread count
CANDLE_Q8K_VALIDATION=1       # Enable validation
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

## Acknowledgments

- [Candle](https://github.com/huggingface/candle) for the excellent ML framework
- [GGML](https://github.com/ggerganov/ggml) for quantization format inspiration

## Citation

If you use this work in research, please cite:

```bibtex
@software{quantize_strategy,
  author = {Ryzhov, Artem},
  title = {Quantize Strategy: Advanced Neural Network Quantization},
  url = {https://github.com/artem1984A/quantize-strategy},
  year = {2024}
}
```