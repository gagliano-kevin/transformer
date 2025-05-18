# NanoTransformer & BPE Tokenizer

This repository contains a minimal yet powerful implementation of the Transformer architecture, known as **NanoTransformer**, along with a custom-built **Byte Pair Encoding (BPE) tokenizer**, both written in pure PyTorch and Python. The goal is to demystify these core NLP components while providing a clean foundation for further research and experimentation.

---

## 📦 Contents

- **NanoTransformer**: Encoder-only Transformer (GPT-style) with:
  - Multi-head self-attention
  - Position embeddings
  - Pre-LayerNorm
  - Residual connections
  - Tied embeddings
- **Custom BPE Tokenizer**:
  - Learns subword vocabulary from scratch
  - Encodes text into token IDs and decodes back
  - Independent of external libraries

---

## 🧠 Transformer Overview

The `transformer` class implements the core Transformer model from scratch with support for:

- Token + Positional Embeddings
- Stacked Transformer blocks
- Multi-head Attention
- MLP with GELU activation
- Weight sharing between input embeddings and output projection
- Configurable model depth, head count, embedding dimensions, and more