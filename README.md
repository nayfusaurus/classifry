# Classifry

[![CI](https://github.com/nayfusaurus/classifry/actions/workflows/build.yml/badge.svg)](https://github.com/nayfusaurus/classifry/actions/workflows/build.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

An email spam classifier powered by a PyTorch neural network. Classifry provides a CLI for training and inference, a Flask web app for reviewing and labeling emails, and a Jupyter notebook for learning the concepts behind it.

## Features

- **Neural network classifier** -- feedforward network with TF-IDF input, trained with early stopping
- **CLI** -- train models, classify emails, and run an interactive REPL
- **Web app** -- upload emails, view a three-panel breakdown (rendered HTML, underlying source, classifier prediction), and label them for retraining
- **Jupyter notebook** -- step-by-step walkthrough of data loading, preprocessing, model building, and evaluation

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

## Getting Started

```bash
git clone https://github.com/nayfusaurus/classifry.git
cd classifry
uv sync
```

Download the [Kaggle dataset](https://www.kaggle.com/datasets/sumit12100012/hamorspam-e-mail-detection-dataset) into the `email-data-set/` directory.

### Train

```bash
uv run python classifier.py train
```

### Classify

```bash
uv run python classifier.py classify --text "Congratulations! You've won a free prize!"
uv run python classifier.py classify path/to/email.txt
```

### Web App

```bash
uv run python app.py
```

Open `http://localhost:5000`. Upload an email file to see the three-panel analysis, then label it as spam or ham to add it to the training dataset.

| Variable | Default | Description |
| --- | --- | --- |
| `FLASK_SECRET_KEY` | dev fallback | Secret key for session management |
| `FLASK_DEBUG` | `true` | Enable debug mode |

## Model

```text
Input (5000 TF-IDF features)
  -> Dense(256) + ReLU + Dropout(0.3)
  -> Dense(128) + ReLU + Dropout(0.3)
  -> Dense(64)  + ReLU + Dropout(0.3)
  -> Dense(1)   + Sigmoid
Output (spam probability 0-1)
```

### Performance

Evaluated on a held-out 20% split (752 emails total):

| | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Ham | 90% | 92% | 91% |
| Spam | 96% | 95% | 95% |

Overall accuracy: **94%**

## CLI Reference

### `train`

```bash
uv run python classifier.py train [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--data-dir` | `email-data-set` | Training data directory |
| `--output-dir` | `models` | Output directory for model artifacts |
| `--epochs` | `50` | Maximum training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--dropout` | `0.3` | Dropout rate |
| `--patience` | `10` | Early stopping patience |

### `classify`

```bash
uv run python classifier.py classify [FILE] [OPTIONS]
```

| Option | Description |
| --- | --- |
| `FILE` | Path to an email file |
| `--text` | Classify text directly instead of a file |
| `--model-dir` | Model directory (default: `models`) |
| `--threshold` | Classification threshold (default: `0.5`) |

### `interactive`

```bash
uv run python classifier.py interactive [OPTIONS]
```

| Option | Description |
| --- | --- |
| `--model-dir` | Model directory (default: `models`) |
| `--threshold` | Classification threshold (default: `0.5`) |

## Project Structure

```text
classifry/
├── classifier.py               # CLI and model code
├── app.py                      # Flask web application
├── spam_classifier_tutorial.ipynb
├── templates/                  # Jinja2 templates
├── tests/                      # pytest suite (118 tests)
├── models/                     # Trained model artifacts
└── email-data-set/             # Training data
```

## Testing

```bash
uv run pytest
uv run pytest -v        # verbose
```

The test suite covers text preprocessing, model inference, Flask routes, CSRF protection, and email parsing.

## Customization

Organize custom training data as:

```text
your-data/
├── ham/
│   └── hard_ham/
└── spam/
    └── spam/
```

Then train:

```bash
uv run python classifier.py train --data-dir your-data
```

## License

Licensed under the [Apache License 2.0](LICENSE).
