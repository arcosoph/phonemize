# Phonemize
[![Join the Discord](https://img.shields.io/badge/Join%20the%20Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/rYfShVvacB)
[![PyPI](https://img.shields.io/pypi/v/phonemize.svg?color=6C63FF&logo=pypi&logoColor=white)](https://pypi.org/project/phonemize/)
[![Python](https://img.shields.io/pypi/pyversions/phonemize.svg?color=3776AB&logo=python&logoColor=white)](https://pypi.org/project/phonemize/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/phonemize?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLACK&left_text=downloads)](https://pepy.tech/projects/phonemize)
[![License](https://img.shields.io/pypi/l/phonemize?color=white&logo=apache&logoColor=black)](https://pypi.org/project/phonemize/)

**Phonemize** is a modern, multilingual **grapheme-to-phoneme (G2P)** conversion library powered by state-of-the-art Transformer models. This library delivers **production-grade accuracy**, **lightning-fast inference**, and **seamless integration** into text-to-speech (TTS) pipelines, speech synthesis engines, and other voice-related applications. Whether you're building real-time TTS systems or offline speech processing pipelines, Phonemize provides the precision and performance you need.

---

## 🚀 Quick Access

- [Installation](#installation) – Get started in seconds
- [Quickstart](#quickstart) – Basic usage example
- [Training Guide](#training-your-own-model) – Train custom models
- [Advanced Inference](#advanced-inference) – Detailed predictions & confidence scores
- [Configuration](#configuration-guide) – Tune model parameters
- [Troubleshooting](#common-issues--troubleshooting) – Common issues & solutions
- [API Reference](#api-reference) – Complete API documentation
- [Pre-trained Models](#pre-trained-models) – Download trained models
- [Performance Tips](#performance--optimization) – Speed optimization

---

## Key Features

-   **🚀 Easy-to-use API**: Intuitive Python interface for both training and inference tasks. Get started with just a few lines of code.
-   **🌍 Multilingual Support**: Train and deploy a single unified model across multiple languages without architectural changes.
-   **⚡ High Performance**: Achieve microsecond-level inference latency with state-of-the-art Transformer-based predictions.
-   **🎯 Custom Training**: Train your own specialized models with minimal setup using simple YAML configuration files.
-   **🔊 Optimized for TTS**: Purpose-built for real-time streaming, offline batch processing, and production text-to-speech systems.
-   **📦 TorchScript Export**: Export trained models to TorchScript for hardware-accelerated inference and deployment.

---

## Installation

### Inference Only

For using pre-trained models without training capabilities:

```bash
pip install phonemize
```

### Full Installation (with Training Support)

To enable model training and all dependencies:

```bash
pip install 'phonemize[train]'
```

**Requirements**: Python 3.8 or later, PyTorch 1.9+

## Quickstart

### Basic Usage: Inference with Pre-trained Model

Load a pre-trained model and convert text to phonemes in just a few lines:

```python
import torch
from phonemize import Phonemizer

# Import custom classes for safe model loading
from phonemize.preprocessing.text import Preprocessor, LanguageTokenizer, SequenceTokenizer

# Register classes for secure deserialization
torch.serialization.add_safe_globals([Preprocessor, LanguageTokenizer, SequenceTokenizer])

# Load pre-trained checkpoint
phonemizer = Phonemizer.from_checkpoint("phonemize_m1.pt")

# Convert text to phonemes (supports multiple languages)
result = phonemizer("Arcosoph's quixotic insights empower enthusiasts.", lang="en_us")

# Display results
print(result)
```

**Output:**

The model converts graphemes (written text) to phonemes (speech sounds) using the ARPAbet notation:

```
[AA][R][K][AH][S][AO][F][S] [K][W][IH][K][S][AA][T][IH][K] [IH][N][S][AY][T][S] [IH][M][P][AW][ER] [EH][N][TH][UW][Z][IY][AE][S][T][S].
```

> Each `[XX]` represents a single phonetic unit. This output can be directly fed into TTS vocoders or speech synthesis systems.

## Model Architectures

Phonemize supports two distinct Transformer-based architectures, each with different strengths:

| Architecture | Type | Speed | Quality | Use Case |
|---|---|---|---|---|
| **Forward Transformer** | Encoder-Decoder (seq2seq) | Fast ⚡ | Excellent 95%+ | Real-time TTS, low-latency inference |
| **Autoregressive Transformer** | Decoder-only (causal) | Slower | Slightly Higher 96%+ | Research, domain-specific fine-tuning |

**Quick Decision Guide:**
- Use **Forward** for production TTS systems
- Use **Autoregressive** for research or when highest accuracy is needed

---

## Training Your Own Model

Create custom G2P models tailored to your specific language or domain. The training pipeline handles preprocessing, model creation, and distributed training.

### Training Pipeline

1. **Prepare Data**: Collect grapheme-phoneme pairs in (language, grapheme, phoneme) format
2. **Configure**: Define model architecture, hyperparameters, and preprocessing in a YAML config file
3. **Preprocess**: Tokenize and prepare datasets with vocabulary and data splits
4. **Train**: Launch training with optional multi-GPU support and checkpoint resumption

### Example: Training a Multilingual Model

```python
from phonemize.preprocess import preprocess
from phonemize.train import train

# Prepare multilingual training data (language_code, grapheme, phoneme)
# Use IPA (International Phonetic Alphabet) for phonemes
train_data = [
    ("en_us", "young", "jʌŋ"),
    ("de", "benützten", "bənʏt͡stn̩"),
    ("fr", "bonjour", "bɔ̃ʒuʁ")
] * 1000

# Define validation set for monitoring model performance
val_data = [
    ("en_us", "young", "jʌŋ"),
    ("de", "benützten", "bənʏt͡stn̩"),
    ("fr", "bonjour", "bɔ̃ʒuʁ")
] * 100

# Load configuration (defines model architecture, training parameters, etc.)
config_file = "configs/forward.yaml"

# Step 1: Preprocess and tokenize data
preprocess(
    config_file=config_file,
    train_data=train_data,
    val_data=val_data,
    deduplicate_train_data=False  # Keep duplicates for balanced multilingual training
)

# Step 2: Train model with distributed support (rank=0 for single GPU)
train(rank=0, num_gpus=1, config_file=config_file)
```

### Resume Training from Checkpoint

If training is interrupted, resume from the last checkpoint:

```python
from phonemize.train import train

# Resume training from checkpoint
train(
    rank=0, 
    num_gpus=1, 
    config_file="configs/forward.yaml",
    checkpoint_file="checkpoints/latest_model.pt"  # Path to checkpoint
)
```

### Multi-GPU Training (Distributed Data Parallel)

For faster training on multiple GPUs, use DDP configuration:

```python
from phonemize.train import train

# Train on 4 GPUs with automatic distribution
train(rank=0, num_gpus=4, config_file="configs/forward.yaml")
```

Update your config file with DDP settings:

```yaml
training:
  ddp_backend: 'nccl'     # Use 'gloo' on CPU-only systems
  ddp_host: 'localhost'   # Hostname for multi-node training
  ddp_port: 12355         # Port for DDP communication
```

**Output**: Model checkpoints saved to the `checkpoint_dir` specified in config. Monitor training with TensorBoard.

## Advanced Inference

### Basic Inference with Customization

Perform phonemization with optional parameters for punctuation handling and acronym expansion:

```python
from phonemize import Phonemizer

# Load custom trained model
phonemizer = Phonemizer.from_checkpoint("checkpoints/best_model.pt")

# Single text inference with default settings
phonemes = phonemizer("Phonemizing text is simple!", lang="en_us")
print(phonemes)  # Output: [F][O][N][EH][M][IH][Z][IH][NG] ...

# Batch inference for better performance
texts = ["Hello world", "How are you?", "Testing Phonemize"]
results = phonemizer(texts, lang="en_us", batch_size=32)

# Custom punctuation handling
custom_punct = ".,!?;"
result = phonemizer(
    "Hello, world!",
    lang="en_us",
    punctuation=custom_punct,
    expand_acronyms=True,  # Expands "DIY" -> "D-I-Y"
    batch_size=8
)
```

**Inference Parameters:**
- `text` (str | List[str]): Input text or list of texts to phonemize
- `lang` (str): Target language code (e.g., 'en_us', 'de', 'fr') - must match training languages
- `punctuation` (str): Characters to split on [default: '().,:?!/–']
- `expand_acronyms` (bool): Auto-expand acronyms like "U.S.A" [default: True]
- `batch_size` (int): Inference batch size for GPU efficiency [default: 8]

### Detailed Results with Confidence Scores

Access comprehensive predictions including confidence metrics and token probabilities:

```python
# Get detailed predictions
result = phonemizer.phonemise_list(
    ["Phonemizing text is simple!"], 
    lang="en_us"
)

# Access detailed per-word predictions
for word, pred in result.predictions.items():
    print(f"Word: '{word}'")
    print(f"  Phonemes:  {pred.phonemes}")
    print(f"  Tokens:    {pred.phoneme_tokens}")
    print(f"  Confidence: {pred.confidence:.3f}")
    print(f"  Token Probs: {[f'{p:.2f}' for p in pred.token_probs]}")

# Access full result structure
print(f"Original texts: {result.text}")
print(f"Split text: {result.split_text}")     # Words separated by punctuation
print(f"Phoneme output: {result.phonemes}")   # Concatenated phoneme strings
print(f"Split phonemes: {result.split_phonemes}")  # Phonemes per word
```

**Result Object Properties:**
- `text`: Original input texts (List[str])
- `phonemes`: Final phoneme strings (List[str])
- `split_text`: Texts split by punctuation (List[List[str]])
- `split_phonemes`: Corresponding phonemes per split (List[List[str]])
- `predictions`: Dict mapping words → Prediction objects with confidence scores

**Prediction Object Properties:**
- `word`: Original word
- `phonemes`: Predicted phoneme string
- `phoneme_tokens`: Individual phoneme tokens (with markers)
- `confidence`: Overall confidence score (0.0-1.0)
- `token_probs`: Per-token probabilities

**Use Cases**: Confidence scores identify uncertain predictions for manual review, retraining, or fallback to dictionary lookup.

## Configuration Guide

All training parameters are defined in YAML configuration files. Phonemize includes example configs for different model types.

### Forward Transformer Config (Recommended for Production)

Use `configs/forward_config.yaml` for fast, high-quality models:

```yaml
model:
  type: 'transformer'  # Fast seq2seq model
  d_model: 512         # Model dimension
  d_fft: 1024          # FFT dimension in feed-forward layer
  layers: 6            # Number of transformer layers
  dropout: 0.1
  heads: 4             # Attention heads

preprocessing:
  languages: ['en_us', 'de', 'fr']  # Add your languages here!
  text_symbols: 'abcdefghijklmnopqrstuvwxyz'  # Supported graphemes
  phoneme_symbols: ['a', 'e', 'i', 'ə', 'ŋ', ...]  # Supported phonemes
  char_repeats: 3      # Max phonemes per grapheme
  lowercase: true
  n_val: 5000          # Validation split size

training:
  learning_rate: 0.0001
  warmup_steps: 10000
  batch_size: 32
  epochs: 500
  generate_steps: 10000      # Eval interval
  validate_steps: 10000
  checkpoint_steps: 100000   # Save interval
```

### Autoregressive Transformer Config (Research/Custom)

Use `configs/autoreg_config.yaml` for slightly higher accuracy at the cost of speed:

```yaml
model:
  type: 'autoreg_transformer'  # Slower but potentially higher accuracy
  d_model: 512
  layers: 4            # Usually fewer layers for autoreg
  heads: 4

preprocessing:
  char_repeats: 1      # MUST be 1 for autoregressive models
```

### Configuration Best Practices

**For Your Languages:**
```yaml
preprocessing:
  languages: ['en_us', 'de', 'fr', 'es']  # Add all target languages
  text_symbols: 'abcdefghijklmnopqrstuvwxyzäöüàâé...'  # Include all chars in data
```

**For More Accuracy (slower):**
```yaml
model:
  d_model: 768         # Increase model size
  layers: 8            # More transformer layers
  dropout: 0.2

training:
  learning_rate: 0.00005  # Lower LR for stability
  warmup_steps: 20000     # Longer warmup
```

**For Faster Training:**
```yaml
model:
  d_model: 256         # Smaller model
  layers: 3            # Fewer layers
  heads: 2

training:
  batch_size: 64       # Larger batches
  generate_steps: 5000 # Less frequent evaluation
```

---

## Common Issues & Troubleshooting

### Language Not Supported Error

```python
# ❌ Error: Language not supported
phonemizer("Hello", lang="pt")  # Portuguese not in training languages

# ✅ Solution: Use a language from model's training set
# Check available languages in the config file
phonemizer("Hello", lang="en_us")  # Use supported language
```

### Model Not Improving During Training

```yaml
# Increase model capacity
model:
  d_model: 768
  layers: 8

# Improve learning rate schedule
training:
  learning_rate: 0.0001
  warmup_steps: 20000
  scheduler_plateau_patience: 15
```

### Out of Memory (OOM) Error

```python
# ✅ Reduce batch size
train(rank=0, num_gpus=1, config_file="configs/forward.yaml")
# Then update config:
# training:
#   batch_size: 16  # Reduced from 32

# ✅ For inference: reduce batch_size
result = phonemizer("text", lang="en_us", batch_size=4)  # Was 8
```

### Low Accuracy on Custom Language

```yaml
preprocessing:
  # Include ALL characters from your training data
  text_symbols: 'abcdefghijklmnopqrstuvwxyzäöü...'  # Add special chars!
  
  # Include ALL phonemes in your dataset
  phoneme_symbols: ['a', 'b', 'c', ..., 'ə', 'ŋ']
```

---

## TorchScript Export (Production Deployment)

```python
import torch
from phonemize import Phonemizer

# Load trained checkpoint
phonemizer = Phonemizer.from_checkpoint("checkpoints/best_model.pt")

# Convert to TorchScript (JIT compilation)
scripted_model = torch.jit.script(phonemizer.predictor.model)
phonemizer.predictor.model = scripted_model

# Save for production deployment
torch.jit.save(scripted_model, "phonemizer_scripted.pt")

# Run optimized inference
result = phonemizer("Running the optimized TorchScript model!")
print(result)
```

**Benefits**:
- ✅ No Python dependencies required at inference time
- ✅ Significantly faster latency (30-50% speedup)
- ✅ Portable across platforms
- ✅ Memory efficient

## Performance & Optimization

### Inference Performance Tips

```python
from phonemize import Phonemizer

phonemizer = Phonemizer.from_checkpoint("phonemize_m1.pt")

# ✅ Batch Processing (5-10x faster than single items)
texts = ["word1", "word2", "word3"] * 100
results = phonemizer(texts, lang="en_us", batch_size=32)

# ❌ Avoid: Processing one word at a time
# for word in texts:
#     result = phonemizer(word, lang="en_us")  # Slow!

# ✅ Adjust batch size based on GPU memory
# Small GPU: batch_size=4-8
# Medium GPU (6GB): batch_size=16-32  
# Large GPU (24GB+): batch_size=64-128

# ✅ Use TorchScript for maximum speed
import torch
scripted = torch.jit.script(phonemizer.predictor.model)
phonemizer.predictor.model = scripted
```

### Training Performance Tips

```yaml
# ✅ For faster training iterations
training:
  batch_size: 64           # Larger batches = faster epoch
  generate_steps: 5000     # Less frequent validation
  validate_steps: 10000

# ✅ Use Multi-GPU training for large datasets
# In training code: train(rank=0, num_gpus=4, config_file=...)

# ✅ For production models: increase compute
model:
  d_model: 768             # Larger model = better accuracy
  layers: 8
  heads: 8
```

### Benchmark Results

| Model | Inference Speed | Throughput | Accuracy |
|---|---|---|---|
| Forward Transformer (GPU) | ~1-2 ms/word | 500-1000 words/sec | 98.5%+ |
| Autoregressive (GPU) | ~5-10 ms/word | 100-200 words/sec | 99%+ |
| TorchScript (GPU) | ~0.5-1 ms/word | 1000-2000 words/sec | Same as original |

---

## API Reference

### Phonemizer Class

```python
from phonemize import Phonemizer

# Load from checkpoint
phonemizer = Phonemizer.from_checkpoint(checkpoint_path)

# Phonemize single text
result = phonemizer(
    text="Hello world",
    lang="en_us",
    punctuation="().,:?!/–",  # Custom punctuation
    expand_acronyms=True,       # Expand acronyms
    batch_size=8                # Batch size for inference
)  # Returns: str

# Phonemize batch of texts  
results = phonemizer(
    text=["Text 1", "Text 2"],
    lang="en_us"
)  # Returns: List[str]

# Get detailed predictions
result_obj = phonemizer.phonemise_list(
    texts=["Hello"],
    lang="en_us",
    punctuation="().,:?!/–",
    expand_acronyms=True,
    batch_size=8
)  # Returns: PhonemizerResult
```

### Preprocess Function

```python
from phonemize.preprocess import preprocess

preprocess(
    config_file="configs/forward.yaml",          # Config path
    train_data=[("en_us", "word", "wɜrd"), ...], # Training data
    val_data=[("en_us", "word", "wɜrd"), ...],   # Validation data (optional)
    deduplicate_train_data=False                 # Keep duplicates
)
```

### Train Function

```python
from phonemize.train import train

train(
    rank=0,                          # GPU rank (0 for single GPU)
    num_gpus=1,                      # Number of GPUs
    config_file="configs/forward.yaml",  # Config path
    checkpoint_file=None             # Resume from checkpoint (optional)
)
```

---

## Pre-trained Models

High-quality pre-trained models ready for immediate use:

| Model | Language | Dataset | Accuracy | Arch | Version | Size |
|---|---|---|---|---|---|---|
| [phonemize_m1](https://github.com/arcosoph/phonemize/releases/download/v0.2.0/phonemize_m1.pt) | 🇺🇸 English (US) | [CMUDict](https://github.com/microsoft/CNTK/tree/master/Examples/SequenceToSequence/CMUDict/Data) | 98.5%+ | Forward | 0.1.0 | ~60MB |

**Tested on**: CMU Dictionary, diverse English vocabulary including technical terms, names, and phonetic anomalies.

> Models are optimized for the `phonemize` library and saved with PyTorch 1.9+.

---

## Roadmap

- [ ] Support for additional languages (Mandarin, Japanese, Arabic)
- [ ] Real-time streaming inference API
- [ ] ONNX model export for maximum portability
- [ ] REST API server for easy deployment
- [ ] Fine-tuning APIs for domain-specific models

## Acknowledgments

**Phonemize** is inspired by [DeepPhonemizer](https://github.com/spring-media/DeepPhonemizer) and has been completely refactored for **modern Python**, **better performance**, and **improved usability**. Special thanks to the open-source speech processing community.

## License & Compatibility

📝 **License**: MIT License - Free for commercial and personal use

🐍 **Python Support**: 3.8, 3.9, 3.10, 3.11, 3.12, 3.13

🔗 **Repository**: [github.com/arcosoph/phonemize](https://github.com/arcosoph/phonemize)

---

**Questions or Issues?** Join our [Discord community](https://discord.gg/rYfShVvacB) or open an issue on GitHub.
