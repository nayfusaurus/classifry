# Email Spam/Ham Classifier

A neural network-based email spam classifier built with PyTorch. This project includes both an educational Jupyter notebook for learning ML concepts and a practical command-line tool.

## Features

- Deep learning spam classifier using PyTorch
- TF-IDF text vectorization
- Interactive Jupyter notebook with detailed explanations
- Command-line interface for training and classification
- GPU support (CUDA) for faster training
- Web interface for reviewing and labeling emails
- Three-panel email analysis (rendered HTML, underlying source, classifier prediction)
- Contribute to training dataset through manual labeling

## Requirements

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd spam-or-ham-classifier
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. Download Kaggle dataset into email-data-set folder.

Kaggle dataset here: [Spam and Ham Dataset](https://www.kaggle.com/datasets/sumit12100012/hamorspam-e-mail-detection-dataset)

## Quick Start

### Train a Model

```bash
uv run python classifier.py train
```

This will:
- Load emails from `email-data-set/`
- Train a neural network classifier
- Save the model to `models/`

### Classify an Email

```bash
# Classify text directly
uv run python classifier.py classify --text "Congratulations! You've won a free prize!"

# Classify a file
uv run python classifier.py classify path/to/email.txt
```

### Interactive Mode

```bash
uv run python classifier.py interactive
```

Enter email text and get instant classification results.

## Web Application

The web application provides a visual interface for reviewing emails and contributing labels to the training dataset.

### Starting the Server

```bash
uv run python app.py
```

The server starts on `http://localhost:5000` by default.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_SECRET_KEY` | (random) | Secret key for session management |
| `FLASK_DEBUG` | `False` | Enable debug mode for development |

Set environment variables before starting:
```bash
export FLASK_SECRET_KEY="your-secret-key"
export FLASK_DEBUG=True
uv run python app.py
```

### Workflow

1. **Browse** - Navigate through emails in the dataset
2. **Review** - View emails in three panels:
   - **Rendered**: HTML content displayed as it would appear in an email client
   - **Underlying**: Raw email source with headers and body
   - **Classifier**: Model prediction with spam probability
3. **Label** - Mark emails as spam or ham to contribute to the training dataset
4. **Train** - Re-train the model with newly labeled data

## Learning with the Notebook

For a detailed walkthrough of how spam classification works:

```bash
uv run jupyter notebook spam_classifier_tutorial.ipynb
```

The notebook covers:
1. **Data Loading** - Parsing raw email files
2. **Text Preprocessing** - Cleaning HTML, removing special characters
3. **Feature Engineering** - TF-IDF vectorization explained
4. **Neural Networks** - Building a classifier from scratch
5. **Training** - Loss functions, optimizers, and backpropagation
6. **Evaluation** - Accuracy, precision, recall, and confusion matrices

## CLI Reference

### Train Command

```bash
uv run python classifier.py train [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | `email-data-set` | Directory containing email data |
| `--output-dir` | `models` | Directory to save trained model |
| `--epochs` | `50` | Maximum training epochs |
| `--batch-size` | `32` | Training batch size |
| `--lr` | `0.001` | Learning rate |
| `--dropout` | `0.3` | Dropout rate for regularization |
| `--patience` | `10` | Early stopping patience |

### Classify Command

```bash
uv run python classifier.py classify [FILE] [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `FILE` | Path to email file to classify |
| `--text` | Email text to classify (instead of file) |
| `--model-dir` | Directory containing trained model (default: `models`) |
| `--threshold` | Classification threshold (default: `0.5`) |

### Interactive Command

```bash
uv run python classifier.py interactive [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--model-dir` | Directory containing trained model (default: `models`) |
| `--threshold` | Classification threshold (default: `0.5`) |

## Project Structure

```
spam-or-ham-classifier/
├── pyproject.toml                  # Project dependencies (uv)
├── classifier.py                   # Command-line tool
├── app.py                          # Flask web application
├── spam_classifier_tutorial.ipynb  # Educational notebook
├── README.md                       # This file
├── templates/                      # Jinja2 HTML templates
│   └── *.html                      # Web interface templates
├── tests/                          # Test suite
│   └── test_*.py                   # Pytest test modules
├── models/                         # Saved models (generated)
│   ├── spam_classifier.pth         # PyTorch model weights
│   └── tfidf_vectorizer.pkl        # TF-IDF vectorizer
└── email-data-set/                 # Training data
    ├── ham/
    │   └── hard_ham/               # Legitimate emails
    └── spam/
        └── spam/                   # Spam emails
```

## Model Architecture

```
Input (5000 TF-IDF features)
    │
    ▼
Dense(256) + ReLU + Dropout(0.3)
    │
    ▼
Dense(128) + ReLU + Dropout(0.3)
    │
    ▼
Dense(64) + ReLU + Dropout(0.3)
    │
    ▼
Dense(1) + Sigmoid
    │
    ▼
Output (spam probability 0-1)
```

## Performance

On the included dataset (752 emails):

| Metric | Ham | Spam |
|--------|-----|------|
| Precision | 90% | 96% |
| Recall | 92% | 95% |
| F1-Score | 91% | 95% |

**Overall Accuracy: 94%**

## Dataset

The project includes a sample dataset with:
- **251 ham emails** (legitimate, "hard" examples)
- **501 spam emails**

Email files are raw RFC 822 format with headers and body.

## Customization

### Using Your Own Data

Organize your emails in the following structure:
```
your-data/
├── ham/
│   └── hard_ham/    # Put legitimate emails here
└── spam/
    └── spam/        # Put spam emails here
```

Then train:
```bash
uv run python classifier.py train --data-dir your-data
```

### Adjusting the Model

Edit `classifier.py` to modify:
- `SpamClassifier` class for architecture changes
- `clean_text()` function for preprocessing changes
- TF-IDF parameters in `train_model()` function

## Running Tests

Run the test suite with pytest:

```bash
uv run pytest
```

For verbose output:

```bash
uv run pytest -v
```

### Test Coverage

The tests cover:

- **Web Application** (`test_app.py`): Flask routes, email rendering, labeling functionality
- **Classifier** (`test_classifier.py`): Text preprocessing, model training, classification accuracy

## Troubleshooting

**"Model files not found"**
- Run `uv run python classifier.py train` first to create a model

**Low accuracy**
- The dataset is small; try collecting more training data
- Adjust dropout rate or learning rate
- Increase `max_features` in TF-IDF vectorizer

**CUDA out of memory**
- Reduce batch size: `--batch-size 16`
- Or use CPU by setting `CUDA_VISIBLE_DEVICES=""`

## License

MIT License
