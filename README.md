# 🛸 NASA Signal Decoder

This project attempts to detect and decrypt a hidden message from noisy signal data.  

---

## 📖 Overview

The decoding process works in three main stages:

1. **Locate the hidden message** by scanning the signal text for the lowest-entropy segment (most structured text).
2. **Evaluate possible decryptions** using a statistical language model (quadgram frequencies).
3. **Crack the cipher key** using simulated annealing to produce the most English-like plaintext.

---

## 📦 Requirements

- Python **3.10+**
- Dependencies listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
