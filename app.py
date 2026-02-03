#!/usr/bin/env python3
"""
Flask Web Application for Email Review and Labeling

A web interface for reviewing emails, viewing classifier predictions,
and labeling emails for dataset expansion.
"""

import hashlib
import os
import re
import secrets
import tempfile
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Comment
from flask import Flask, flash, redirect, render_template, request, session, url_for

from classifier import EmailClassifier, clean_text, parse_email_file

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-only-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'.eml', '.txt', '.msg'}

# Configuration
MODEL_DIR = Path('models')
DATA_DIR = Path('email-data-set')
HAM_DIR = DATA_DIR / 'ham' / 'hard_ham'
SPAM_DIR = DATA_DIR / 'spam' / 'spam'


# =============================================================================
# Helper Functions
# =============================================================================

def safe_render_html(html: str) -> str:
    """
    Safely render HTML by stripping dangerous elements and disabling links.

    - Removes script, style, iframe, object, embed tags
    - Removes event handlers (onclick, onload, etc.)
    - Disables form elements
    - Converts links to non-clickable text
    """
    if not html:
        return ''

    soup = BeautifulSoup(html, 'html.parser')

    # Remove dangerous tags entirely
    dangerous_tags = ['script', 'style', 'iframe', 'object', 'embed', 'frame',
                      'frameset', 'meta', 'link', 'base']
    for tag in soup.find_all(dangerous_tags):
        tag.decompose()

    # Remove HTML comments (can contain conditional IE scripts)
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove event handlers from all tags
    event_attrs = [attr for attr in ['onclick', 'onload', 'onerror', 'onmouseover',
                                      'onmouseout', 'onfocus', 'onblur', 'onsubmit',
                                      'onchange', 'onkeydown', 'onkeyup', 'onkeypress']]
    for tag in soup.find_all(True):
        for attr in list(tag.attrs.keys()):
            # Remove event handlers
            if attr.lower().startswith('on'):
                del tag[attr]
            # Remove javascript: URLs
            if attr.lower() in ['href', 'src', 'action'] and tag.get(attr, '').lower().startswith('javascript:'):
                del tag[attr]

    # Disable forms
    for form in soup.find_all('form'):
        form['action'] = '#'
        form['onsubmit'] = 'return false;'

    # Disable all input elements
    for inp in soup.find_all(['input', 'button', 'select', 'textarea']):
        inp['disabled'] = 'disabled'

    # Convert links to span elements (preserve text, remove clickability)
    for a in soup.find_all('a'):
        span = soup.new_tag('span')
        span['class'] = 'disabled-link'
        span['title'] = f"Link: {a.get('href', 'unknown')}"
        span['style'] = 'color: #666; text-decoration: underline; cursor: not-allowed;'
        span.string = a.get_text() or a.get('href', '[link]')
        a.replace_with(span)

    return str(soup)


def extract_underlying_content(email_text: str) -> dict:
    """
    Extract underlying content from email for security analysis.

    Returns:
        dict with:
        - urls: List of all URLs found
        - hidden_text: Text that might be hidden (white text, tiny font, etc.)
        - link_mismatches: Links where display text differs from actual URL
        - form_actions: Any form action URLs
        - suspicious_elements: Other suspicious patterns
    """
    result = {
        'urls': [],
        'hidden_text': [],
        'link_mismatches': [],
        'form_actions': [],
        'suspicious_elements': []
    }

    if not email_text:
        return result

    # Extract all URLs using regex
    url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
    result['urls'] = list(set(re.findall(url_pattern, email_text, re.IGNORECASE)))

    # Parse as HTML for deeper analysis
    soup = BeautifulSoup(email_text, 'html.parser')

    # Find link mismatches (display text vs actual URL)
    for a in soup.find_all('a', href=True):
        href = a.get('href', '')
        display_text = a.get_text().strip()

        # Check if display text looks like a URL but differs from href
        if display_text and (display_text.startswith('http') or display_text.startswith('www.')):
            # Normalize for comparison
            display_domain = urlparse(display_text if display_text.startswith('http') else f'http://{display_text}').netloc
            href_domain = urlparse(href if href.startswith('http') else f'http://{href}').netloc

            if display_domain and href_domain and display_domain.lower() != href_domain.lower():
                result['link_mismatches'].append({
                    'display': display_text,
                    'actual': href,
                    'display_domain': display_domain,
                    'actual_domain': href_domain
                })

    # Find hidden text (common spam techniques)
    for tag in soup.find_all(True):
        style = tag.get('style', '').lower()

        # Check for hidden content indicators
        hidden_indicators = [
            'display:none', 'display: none',
            'visibility:hidden', 'visibility: hidden',
            'font-size:0', 'font-size: 0',
            'font-size:1px', 'font-size: 1px',
            'color:#fff', 'color: #fff', 'color:white', 'color: white',
            'height:0', 'height: 0',
            'width:0', 'width: 0',
            'opacity:0', 'opacity: 0'
        ]

        for indicator in hidden_indicators:
            if indicator in style:
                text = tag.get_text().strip()
                if text and len(text) > 2:
                    result['hidden_text'].append({
                        'text': text[:200],
                        'reason': indicator
                    })
                break

    # Find form actions
    for form in soup.find_all('form', action=True):
        action = form.get('action', '')
        if action and action != '#':
            result['form_actions'].append(action)

    # Find suspicious patterns
    suspicious_patterns = [
        (r'<img[^>]*src=["\'][^"\']*width=["\']?1["\']?[^>]*height=["\']?1["\']?', 'Tracking pixel detected'),
        (r'<img[^>]*height=["\']?1["\']?[^>]*width=["\']?1["\']?', 'Tracking pixel detected'),
        (r'base64,', 'Base64 encoded content'),
        (r'data:image', 'Embedded data URI image'),
    ]

    for pattern, description in suspicious_patterns:
        if re.search(pattern, email_text, re.IGNORECASE):
            if description not in result['suspicious_elements']:
                result['suspicious_elements'].append(description)

    return result


def generate_unique_filename() -> str:
    """
    Generate a unique filename for user-labeled emails.

    Format: user_{timestamp}_{hash}.txt
    """
    timestamp = int(time.time() * 1000)  # Millisecond precision
    unique_string = f"{timestamp}-{time.time_ns()}"
    hash_suffix = hashlib.md5(unique_string.encode()).hexdigest()[:8]
    return f"user_{timestamp}_{hash_suffix}.txt"


def get_classifier() -> Optional[EmailClassifier]:
    """
    Get the email classifier instance.
    Returns None if model is not available.
    """
    try:
        return EmailClassifier(MODEL_DIR)
    except FileNotFoundError:
        return None


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    if not filename:
        return False
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def generate_csrf_token() -> str:
    """Generate a CSRF token and store it in the session."""
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(32)
    return session['csrf_token']


def validate_csrf_token() -> bool:
    """Validate the CSRF token from the form against the session."""
    token = request.form.get('csrf_token', '')
    return token and token == session.get('csrf_token')


# Make csrf_token available in all templates
app.jinja_env.globals['csrf_token'] = generate_csrf_token


# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def index():
    """Main page with upload form."""
    # Check if model is available
    model_available = (MODEL_DIR / 'spam_classifier.pth').exists()
    return render_template('index.html', model_available=model_available)


@app.route('/upload', methods=['POST'])
def upload():
    """
    Handle email file upload.

    - Parse the uploaded email
    - Prepare rendered view, underlying content, and classifier view
    - Classify the email
    - Return results page
    """
    # Validate CSRF token
    if not validate_csrf_token():
        flash('Invalid request. Please try again.', 'error')
        return redirect(url_for('index'))

    if 'email_file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))

    file = request.files['email_file']

    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))

    # Validate file extension
    if not allowed_file(file.filename):
        flash('Invalid file type. Only .eml, .txt, and .msg files are allowed.', 'error')
        return redirect(url_for('index'))

    # Save to temporary file and parse
    try:
        # Create a temporary file to parse
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.eml', delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)

        # Parse the email
        parsed = parse_email_file(tmp_path)

        # Read raw content for analysis
        with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_content = f.read()

        # Clean up temp file
        tmp_path.unlink()

    except Exception as e:
        flash(f'Error parsing email: {str(e)}', 'error')
        return redirect(url_for('index'))

    # Prepare the three views
    email_body = parsed.get('body', '')
    email_subject = parsed.get('subject', '')
    combined_text = f"{email_subject} {email_body}"

    # 1. Rendered view (safe HTML)
    rendered_view = safe_render_html(email_body)

    # 2. Underlying content analysis
    underlying = extract_underlying_content(raw_content)

    # 3. Classifier view (cleaned text)
    classifier_view = clean_text(combined_text)

    # Classify the email
    classifier = get_classifier()
    if classifier:
        classification = classifier.classify(combined_text)
    else:
        classification = {
            'prediction': 'UNKNOWN',
            'spam_probability': 0.5,
            'confidence': 0.0,
            'error': 'Model not trained yet'
        }

    return render_template(
        'result.html',
        subject=email_subject,
        rendered_view=rendered_view,
        underlying=underlying,
        classifier_view=classifier_view,
        raw_body=email_body,
        classification=classification,
        original_filename=file.filename
    )


@app.route('/label', methods=['POST'])
def label():
    """
    Save email with user's label to the dataset.

    - Accept label (spam/ham) and email content
    - Generate unique filename
    - Save to appropriate directory
    """
    # Validate CSRF token
    if not validate_csrf_token():
        flash('Invalid request. Please try again.', 'error')
        return redirect(url_for('index'))

    label_value = request.form.get('label')
    email_content = request.form.get('email_content', '')

    if not label_value or label_value not in ['spam', 'ham']:
        flash('Invalid label. Please select spam or ham.', 'error')
        return redirect(url_for('index'))

    if not email_content.strip():
        flash('No email content to save.', 'error')
        return redirect(url_for('index'))

    # Determine save directory
    if label_value == 'ham':
        save_dir = HAM_DIR
    else:
        save_dir = SPAM_DIR

    # Ensure directory exists
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    filename = generate_unique_filename()
    filepath = save_dir / filename

    # Save the cleaned text content
    try:
        cleaned_content = clean_text(email_content)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

        flash(f'Email saved as {label_value} to {filepath.name}', 'success')
    except Exception as e:
        flash(f'Error saving email: {str(e)}', 'error')

    return redirect(url_for('index'))


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    # Ensure required directories exist
    MODEL_DIR.mkdir(exist_ok=True)
    HAM_DIR.mkdir(parents=True, exist_ok=True)
    SPAM_DIR.mkdir(parents=True, exist_ok=True)

    app.run(debug=os.environ.get('FLASK_DEBUG', 'true').lower() == 'true')
