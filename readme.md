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

---

## 📜 Custom BPE Tokenizer Overview

A Python implementation of a Byte Pair Encoding (BPE) tokenizer. This tokenizer can be trained on a given text corpus to learn a vocabulary and merge rules. It can then be used to encode text into a sequence of tokens and decode tokens back into text.

## Features

-   **Trainable:** Learn BPE merge rules and vocabulary from raw text or text files.
-   **Configurable Vocabulary Size:** Specify the desired vocabulary size during initialization.
-   **Encoding & Decoding:** Convert text to token IDs and vice-versa.
-   **Model Persistence:** Save trained tokenizer (vocabulary and merge rules) to files and load them later.
-   **Vocabulary Display:** Inspect the learned vocabulary with human-readable representations of tokens, including proper handling of control characters and extended ASCII.
-   **Logging:** Optional logging for training and encoding steps.
-   **Unicode Handling:** Properly handles Unicode characters, including replacement of control characters for display purposes.
