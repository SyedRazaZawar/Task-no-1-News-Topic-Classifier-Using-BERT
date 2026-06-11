# News Topic Classifier Using BERT

An end-to-end Natural Language Processing (NLP) application that fine-tunes a transformer model (BERT) to classify news headlines into four categories: **World**, **Sports**, **Business**, and **Sci/Tech** using the Hugging Face library, and deploys it on a premium interactive Streamlit dashboard.

---

## Features
- **Transformers Training Loop:** Utilizes Hugging Face `Trainer` API to tokenize, train, and evaluate `bert-base-uncased` (or alternative models) on the AG News dataset.
- **Robust Evaluation:** Reports validation loss, Accuracy, and Weighted F1-score.
- **Adaptive Training:** Supports training on configurable subset sizes (`--subset_size`) for fast development runs on CPU.
- **Glassmorphism Web Dashboard:** A dark-themed, responsive web UI built with Streamlit featuring:
  - Interactive headline classification.
  - Dynamic probability distributions using beautiful custom status progress bars.
  - Dynamic model loader with support for local trained weights or direct Hugging Face Hub inference fallback (`mrm8488/bert-mini-finetuned-ag_news`).
  - Automatically parses training states to display performance metrics in the sidebar.

---

## Directory Structure
```
├── requirements.txt      # Project library dependencies
├── train.py              # CLI training and evaluation script
├── app.py                # Streamlit Web App code
└── README.md             # Documentation
```

---

## Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed (tested on Python 3.14).

### 2. Setup Virtual Environment
Create and activate a virtual environment to isolate project dependencies:

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using pip:
```bash
pip install -r requirements.txt
```

---

## Training the Model

The training script `train.py` downloads the AG News dataset from Hugging Face Hub, tokenizes the text, fine-tunes the transformer model, and saves the best model in `./results/best_model`.

### Running on a GPU (Full Training)
If you have an NVIDIA GPU (with CUDA setup), run the full training using:
```bash
python train.py --model_name bert-base-uncased --epochs 3 --batch_size 16
```

### Running on a CPU (Fast Testing / Subset)
Fine-tuning a standard BERT model on the full 120,000 sample dataset on a CPU is very slow. To run a fast test or train on a CPU, use a subset of the dataset and a smaller model (such as DistilBERT or TinyBERT):

```bash
# Trains a small model on a subset of 1000 samples for 1 epoch
python train.py --model_name prajjwal1/bert-tiny --epochs 1 --batch_size 8 --subset_size 1000
```

### Training CLI Arguments:
- `--model_name`: Hugging Face model identifier (default: `bert-base-uncased`).
- `--epochs`: Number of training epochs (default: `3`).
- `--batch_size`: Batch size for training and evaluation (default: `16`).
- `--subset_size`: Number of training samples to use (default: `0` for full dataset).
- `--lr`: Learning rate (default: `2e-5`).
- `--save_dir`: Directory to save the final fine-tuned model (default: `./results/best_model`).

---

## Running the Web App

Start the interactive Streamlit dashboard:
```bash
streamlit run app.py
```

### Fallback Mode
If you run the app before training a local model, it will display a warning and automatically download a lightweight, pre-trained AG News classifier (`mrm8488/bert-mini-finetuned-ag_news`) from the Hugging Face Hub so you can interact with the classifier immediately.

---

## Verification & Examples
Input sample news topics into the app to see predictions:
* **World:** *"UN diplomats call for ceasefire negotiations in Eastern Europe"*
* **Sports:** *"Striker hits crucial goal in stoppage time to win Championship"*
* **Business:** *"Global stock markets tumble as inflation concerns re-emerge"*
* **Sci/Tech:** *"Astronomers discover a new habitable exoplanet using telescope arrays"*
