#!/usr/bin/env python3
"""
Classifry - Email Spam Classifier

A command-line tool for training and using a neural network spam classifier.

Usage:
    # Train a new model
    python classifier.py train --data-dir email-data-set

    # Classify a single email file
    python classifier.py classify email.txt

    # Classify text directly
    python classifier.py classify --text "Your email content here"

    # Interactive mode
    python classifier.py interactive
"""

import argparse
import pickle
import re
import sys
from email import policy
from pathlib import Path
from typing import Optional

import email as email_lib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# =============================================================================
# Model Definition
# =============================================================================

class SpamClassifier(nn.Module):
    """
    Feedforward neural network for spam classification.
    """

    def __init__(self, input_dim: int, dropout_rate: float = 0.3):
        super(SpamClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# =============================================================================
# Text Processing
# =============================================================================

def clean_text(text: str) -> str:
    """Clean and preprocess email text."""
    if not text or not isinstance(text, str):
        return ''

    # Remove HTML tags
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text(separator=' ')

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' url ', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' emailaddr ', text)

    # Remove numbers
    text = re.sub(r'\b\d+\b', ' ', text)

    # Keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def parse_email_file(file_path: Path) -> dict:
    """Parse an email file and extract content."""
    try:
        for encoding in ['utf-8', 'latin-1', 'ascii', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    raw_content = f.read()
                break
            except UnicodeDecodeError:
                continue

        msg = email_lib.message_from_string(raw_content, policy=policy.default)

        subject = str(msg.get('Subject', '') or '')

        body = ''
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    try:
                        body = part.get_content()
                        break
                    except:
                        pass
                elif content_type == 'text/html' and not body:
                    try:
                        body = part.get_content()
                    except:
                        pass
        else:
            try:
                body = msg.get_content()
            except:
                body = raw_content

        return {'subject': subject, 'body': str(body) if body else ''}
    except Exception:
        return {'subject': '', 'body': raw_content if 'raw_content' in dir() else ''}


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset(data_dir: Path) -> tuple:
    """Load emails from directory structure."""
    ham_dir = data_dir / "ham" / "hard_ham"
    spam_dir = data_dir / "spam" / "spam"

    if not ham_dir.exists() or not spam_dir.exists():
        raise FileNotFoundError(
            f"Expected directories:\n  {ham_dir}\n  {spam_dir}\n"
            "Please check your data directory structure."
        )

    texts = []
    labels = []

    # Load ham emails
    ham_files = [f for f in ham_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
    print(f"Loading {len(ham_files)} ham emails...")
    for file_path in tqdm(ham_files, desc="Ham"):
        parsed = parse_email_file(file_path)
        combined = f"{parsed['subject']} {parsed['body']}"
        cleaned = clean_text(combined)
        if cleaned:
            texts.append(cleaned)
            labels.append(0)

    # Load spam emails
    spam_files = [f for f in spam_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
    print(f"Loading {len(spam_files)} spam emails...")
    for file_path in tqdm(spam_files, desc="Spam"):
        parsed = parse_email_file(file_path)
        combined = f"{parsed['subject']} {parsed['body']}"
        cleaned = clean_text(combined)
        if cleaned:
            texts.append(cleaned)
            labels.append(1)

    return texts, labels


# =============================================================================
# Training
# =============================================================================

def train_model(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.3,
    patience: int = 10
) -> dict:
    """Train a new spam classifier model."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\n" + "=" * 50)
    print("Loading dataset...")
    print("=" * 50)
    texts, labels = load_dataset(data_dir)
    print(f"\nTotal emails: {len(texts)}")
    print(f"Ham: {labels.count(0)}, Spam: {labels.count(1)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Vectorize
    print("\nVectorizing text...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_tfidf.toarray()).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_tensor = torch.FloatTensor(X_test_tfidf.toarray()).to(device)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_dim = X_train_tfidf.shape[1]
    model = SpamClassifier(input_dim=input_dim, dropout_rate=dropout_rate).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Training loop
    print("\n" + "=" * 50)
    print("Training...")
    print("=" * 50)

    best_test_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Evaluate
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                test_loss += criterion(outputs, batch_y).item()
                preds = (outputs >= 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        test_loss /= len(test_loader)
        test_acc = correct / total

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_acc:.4f}")

        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Final evaluation
    print("\n" + "=" * 50)
    print("Final Evaluation")
    print("=" * 50)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            preds = (outputs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(batch_y.cpu().numpy().flatten())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Ham', 'Spam']))

    # Save model
    output_dir.mkdir(exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'dropout_rate': dropout_rate
    }, output_dir / 'spam_classifier.pth')

    with open(output_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    print(f"\nModel saved to {output_dir}/")

    return {
        'accuracy': correct / total,
        'model_path': output_dir / 'spam_classifier.pth',
        'vectorizer_path': output_dir / 'tfidf_vectorizer.pkl'
    }


# =============================================================================
# Classification
# =============================================================================

class EmailClassifier:
    """Wrapper class for loading and using trained models."""

    def __init__(self, model_dir: Path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        model_path = model_dir / 'spam_classifier.pth'
        vectorizer_path = model_dir / 'tfidf_vectorizer.pkl'

        if not model_path.exists() or not vectorizer_path.exists():
            raise FileNotFoundError(
                f"Model files not found in {model_dir}. "
                "Please train a model first with 'classifier.py train'"
            )

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model = SpamClassifier(
            input_dim=checkpoint['input_dim'],
            dropout_rate=checkpoint.get('dropout_rate', 0.3)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        with open(vectorizer_path, 'rb') as f:
            self.tfidf = pickle.load(f)

    def classify(self, text: str, threshold: float = 0.5) -> dict:
        """Classify email text."""
        cleaned = clean_text(text)

        if not cleaned:
            return {
                'prediction': 'UNKNOWN',
                'spam_probability': 0.5,
                'confidence': 0.0,
                'error': 'Empty text after cleaning'
            }

        features = self.tfidf.transform([cleaned]).toarray()
        features_tensor = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            probability = self.model(features_tensor).item()

        return {
            'prediction': 'SPAM' if probability >= threshold else 'HAM',
            'spam_probability': probability,
            'confidence': max(probability, 1 - probability)
        }

    def classify_file(self, file_path: Path, threshold: float = 0.5) -> dict:
        """Classify an email file."""
        parsed = parse_email_file(file_path)
        combined = f"{parsed['subject']} {parsed['body']}"
        result = self.classify(combined, threshold)
        result['subject'] = parsed['subject'][:100] if parsed['subject'] else '(no subject)'
        return result


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_train(args):
    """Train command handler."""
    train_model(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout_rate=args.dropout,
        patience=args.patience
    )


def cmd_classify(args):
    """Classify command handler."""
    classifier = EmailClassifier(Path(args.model_dir))

    if args.text:
        result = classifier.classify(args.text, args.threshold)
    elif args.file:
        result = classifier.classify_file(Path(args.file), args.threshold)
        if 'subject' in result:
            print(f"Subject: {result['subject']}")
    else:
        print("Error: Provide either --text or a file path")
        sys.exit(1)

    print(f"Prediction: {result['prediction']}")
    print(f"Spam Probability: {result['spam_probability']:.2%}")
    print(f"Confidence: {result['confidence']:.2%}")


def cmd_interactive(args):
    """Interactive mode command handler."""
    print("Loading model...")
    classifier = EmailClassifier(Path(args.model_dir))

    print("\n" + "=" * 50)
    print("Classifry - Interactive Mode")
    print("=" * 50)
    print("Enter email text (end with an empty line) or 'quit' to exit.\n")

    while True:
        print("-" * 30)
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break

            if line.lower() == 'quit':
                print("Goodbye!")
                return

            if line == '' and lines:
                break

            lines.append(line)

        if not lines:
            continue

        text = '\n'.join(lines)
        result = classifier.classify(text, args.threshold)

        print(f"\nResult: {result['prediction']}")
        print(f"Spam Probability: {result['spam_probability']:.2%}")
        print(f"Confidence: {result['confidence']:.2%}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Classifry - Email Spam Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data-dir', default='email-data-set',
                             help='Directory containing email data')
    train_parser.add_argument('--output-dir', default='models',
                             help='Directory to save model')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Training batch size')
    train_parser.add_argument('--lr', type=float, default=0.001,
                             help='Learning rate')
    train_parser.add_argument('--dropout', type=float, default=0.3,
                             help='Dropout rate')
    train_parser.add_argument('--patience', type=int, default=10,
                             help='Early stopping patience')
    train_parser.set_defaults(func=cmd_train)

    # Classify command
    classify_parser = subparsers.add_parser('classify', help='Classify an email')
    classify_parser.add_argument('file', nargs='?', help='Email file to classify')
    classify_parser.add_argument('--text', help='Email text to classify')
    classify_parser.add_argument('--model-dir', default='models',
                                help='Directory containing trained model')
    classify_parser.add_argument('--threshold', type=float, default=0.5,
                                help='Classification threshold')
    classify_parser.set_defaults(func=cmd_classify)

    # Interactive command
    interactive_parser = subparsers.add_parser('interactive',
                                               help='Interactive classification mode')
    interactive_parser.add_argument('--model-dir', default='models',
                                   help='Directory containing trained model')
    interactive_parser.add_argument('--threshold', type=float, default=0.5,
                                   help='Classification threshold')
    interactive_parser.set_defaults(func=cmd_interactive)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
