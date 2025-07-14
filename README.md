# Text-Classification

A comprehensive text classification project using the IMDB dataset to perform sentiment analysis with three different deep learning models.

## Dataset

We use the Stanford IMDB dataset from Hugging Face: [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)

- Training set: 25,000 movie reviews
- Test set: 25,000 movie reviews
- Labels: 0 (negative), 1 (positive)

## Model Architectures

This project implements three different models for text classification:

### 1. TextCNN
- Multiple convolutional kernels of different sizes (3, 4, 5)
- Word embedding dimension: configurable
- Number of filters: configurable
- Dropout layers to prevent overfitting

### 2. LSTM
- Bidirectional LSTM network
- Word embedding dimension: configurable
- Hidden layer dimension: configurable
- Number of layers: configurable
- Dropout layers

### 3. BERT
- Uses pre-trained bert-base-uncased model (or other HuggingFace BERT models)
- Adds a classification head on top of BERT
- Supports fine-tuning

## Project Structure

```
Text-Classification/
├── main.py              # Main execution script
├── config.py            # Configuration file
├── data_processor.py    # Data processing module
├── models/              # Model definitions (TextCNN, LSTM, BERT)
├── train.py             # Training script
├── requirements.txt     # Dependencies
├── data/                # Data directory
├── models/              # Saved models
├── output/              # Output results
├── logs/                # Training logs
└── cache/               # Cache directory
```

## Installation and Usage

### 1. Environment Setup

```bash
# Clone the project
git clone <repository-url>
cd Text-Classification

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Training Models

You can train any of the three models (TextCNN, LSTM, BERT) using the training script. The IMDB dataset will be automatically downloaded and preprocessed.

#### Train all models (TextCNN, LSTM, BERT):
```bash
python train.py
```

#### Train a specific model (e.g., LSTM):
```bash
python train.py --model lstm
```

#### Train with custom parameters:
```bash
python train.py --model bert --epochs 10 --batch_size 16 --lr 3e-5
```

#### Disable Weights & Biases logging:
```bash
python train.py --model textcnn --no_wandb
```

### 3. Configuration

Main configurations are in `config.py`:

- `DATA_CONFIG`: Data-related configurations (batch size, sequence length, etc.)
- `MODEL_CONFIG`: Model-related configurations (vocabulary size, embedding dimension, etc.)
- `TRAINING_CONFIG`: Training-related configurations (learning rate, epochs, gradient accumulation, etc.)

You can modify these settings to suit your hardware and experiment needs.

### 4. Output Results

After training, the following files will be generated:

- `models/`: Saved model checkpoints
- `output/`: Training curves and comparison charts
- `logs/`: Training logs

## Features

- Supports three model architectures: TextCNN, LSTM, and BERT
- Easy switching between models via command line
- Flexible configuration for model and training parameters
- Automatic download and preprocessing of the IMDB dataset
- Training progress logging with TensorBoard and Weights & Biases (wandb)
- Model checkpoints and training curves saved for later analysis

## Model Performance (Reference)

| Model   | Accuracy | F1-Score | Parameters |
|---------|----------|----------|------------|
| TextCNN | ~87%     | ~87%     | ~2.5M      |
| LSTM    | ~85%     | ~85%     | ~3.2M      |
| BERT    | ~93%     | ~93%     | ~110M      |

---

For more details on configuration and advanced usage, please refer to the comments in `config.py` and the source code.


