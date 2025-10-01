# Bowel-sounds-classification

This repository provides a **pipeline for bowel sound classification** using state-of-the-art deep learning models.  
The pipeline is implemented with **PyTorch**, **Hugging Face Transformers**, **Torchaudio**, and **Scikit-Learn**.  
It supports **single-model fine-tuning** as well as **ensemble learning** across multiple pre-trained speech models.

---

## Project Overview

- **Input**
  - Raw audio files (`.wav`) containing bowel sound recordings.
  - Ground-truth `.txt` files with start time, end time, and label of each sub-audio segment.

- **Pipeline Features**
  - Fine-tuning of multiple Hugging Face models (e.g., HuBERT, Wav2Vec2, WavLM).
  - Support for **imbalanced datasets** (downsampling/upsampling).
  - Evaluation with **precision, recall, F1-score** (per class).
  - Saving of **confusion matrices** for deeper analysis.
  - Ensemble prediction (average of logits across models).

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/bowel-sound-classification.git
cd bowel-sound-classification
pip install -r requirements.txt
````


## Usage

### 1. Training

To fine-tune a model (or an ensemble of models):

```bash
python train.py 
```
### This will:

- Load the dataset defined in `config.yaml`.  
- Perform train/validation/test split.  
- Fine-tune each model (or ensemble).  
- Save the best model checkpoint and logs in `./results`.

## 2. Inference & Evaluation

To run inference on test audio and compare with ground truth:

```bash
python inference.py
```
### This will:

- Load the trained model(s).
- Perform sliding-window multi-event prediction.  
- Merge consecutive predictions of the same class. 
- Evaluate predictions against ground-truth `.txt`.
- Compute IoU-based matching with metrics: precision, recall, F1-score.
- Save a detailed CSV report and confusion matrix plots.
