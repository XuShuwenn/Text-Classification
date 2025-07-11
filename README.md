# Text-Classification

A comprehensive text classification project using the IMDB dataset to perform sentiment analysis with three different deep learning models.

## Dataset

We use the Stanford IMDB dataset from Hugging Face: [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)

- Training set: 25,000 movie reviews
- Test set: 25,000 movie reviews
- Labels: 0 (negative), 1 (positive)

## Model Architectures

This project implements three different models:

### 1. TextCNN
- Uses multiple convolutional kernels of different sizes (3, 4, 5)
- Word embedding dimension: 100
- Number of filters: 100
- Includes Dropout layers to prevent overfitting

### 2. LSTM
- Bidirectional LSTM network
- Word embedding dimension: 100
- Hidden layer dimension: 128
- Number of layers: 2
- Includes Dropout layers

### 3. BERT
- Uses pre-trained bert-base-uncased model
- Adds classification head on top of BERT
- Supports fine-tuning

## Project Structure

```
Text-Classification/
├── main.py              # Main execution script
├── config.py            # Configuration file
├── data_processor.py    # Data processing module
├── models.py            # Model definitions
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── inference.py        # Inference script
├── requirements.txt    # Dependencies
├── data/              # Data directory
├── models/            # Saved models
├── output/            # Output results
├── logs/              # Training logs
└── cache/             # Cache directory
```

## Installation and Usage

### 1. Environment Setup

```bash
# Clone the project
git clone <repository-url>
cd Text-Classification

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Quick Start

```bash
# Check environment
python main.py check

# Setup project directories
python main.py setup

# Train all models
python main.py train

# Evaluate models
python main.py evaluate

# Inference (interactive mode)
python main.py infer --model bert --interactive

# Inference (single prediction)
python main.py infer --model bert --text "This movie is amazing!"
```

### 3. Detailed Usage

#### Training Models
```bash
# Train all models (TextCNN, LSTM, BERT)
python train.py

# Or use main script
python main.py train
```

#### Evaluating Models
```bash
# Evaluate all trained models
python evaluate.py

# Or use main script
python main.py evaluate
```

#### Inference
```bash
# Use BERT model for inference
python inference.py --model bert --text "Great movie!"

# Interactive mode
python inference.py --model bert --interactive

# Use other models
python inference.py --model textcnn --text "Terrible film!"
python inference.py --model lstm --text "Amazing story!"
```

## Configuration

Main configurations are in `config.py`:

- `DATA_CONFIG`: Data-related configurations (batch size, sequence length, etc.)
- `MODEL_CONFIG`: Model-related configurations (vocabulary size, embedding dimension, etc.)
- `TRAINING_CONFIG`: Training-related configurations (learning rate, epochs, etc.)

## Output Results

After training and evaluation, the following files will be generated:

- `models/`: Saved model checkpoints
- `output/model_comparison.png`: Model performance comparison chart
- `output/all_confusion_matrices.png`: Confusion matrices
- `output/evaluation_report.txt`: Detailed evaluation report
- `logs/`: TensorBoard log files

## Model Performance

Typical performance (for reference only):

| Model | Accuracy | F1-Score | Parameters |
|-------|----------|----------|------------|
| TextCNN | ~87% | ~87% | ~2.5M |
| LSTM | ~85% | ~85% | ~3.2M |
| BERT | ~93% | ~93% | ~110M |


