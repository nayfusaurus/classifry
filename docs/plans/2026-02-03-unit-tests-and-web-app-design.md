# Unit Tests & Web App Design

**Status: IMPLEMENTED** (2026-02-03)

## Overview

This document covers two enhancements to the spam-or-ham-classifier project:

1. **Unit tests** with CI integration
2. **Web application** for reviewing and labeling emails

---

## Part 1: Unit Tests

### Test Coverage

| Module | Tests |
|--------|-------|
| `clean_text()` | HTML removal, URL replacement, email sanitization, empty input handling |
| `parse_email_file()` | Plain text emails, multipart MIME, encoding fallbacks, malformed files |
| `SpamClassifier` | Model initialization, forward pass shape, output range (0-1) |
| `EmailClassifier` | Loading saved models, classification output structure, threshold behavior |

### Test Framework

- **pytest** - standard Python testing framework
- Location: `tests/test_classifier.py`

### Test Fixtures

- `tests/fixtures/sample_ham.txt` - legitimate email sample
- `tests/fixtures/sample_spam.txt` - spam email sample
- `tests/fixtures/sample_html.eml` - HTML email with hidden content

---

## Part 2: CI Workflow Fixes

### Current Issues

1. Duplicate "Install dependencies" step names (lines 28-32)
2. Bandit only scans `classifier.py`, should scan all Python files
3. No test execution step

### Fixed Workflow

```yaml
name: Spam Ham Classifier

on:
  push:
    branches: [ "master" ]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v5

      - name: Set up Python
        uses: astral-sh/setup-uv@v7
        with:
          python-version: '3.11'
          enable-cache: true
          cache-dependency-glob: |
            **/pyproject.toml
            **/uv.lock

      - name: Install dependencies
        run: uv sync --dev

      - name: Run tests
        run: uv run pytest tests/ -v

      - name: Run Bandit security scan
        run: uv run bandit -r . --exclude tests,docs -q
```

---

## Part 3: Web Application

### Purpose

A Flask web app that allows users to:
1. Upload email files
2. View sanitized content alongside rendered HTML view
3. See model prediction with confidence
4. Label emails as spam/ham
5. Add labeled emails to the training dataset

### Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Main page with upload form |
| `/upload` | POST | Receive email file, return sanitized view + prediction |
| `/label` | POST | Save email to dataset with user's label |

### Three-Panel Display

The result page shows three views of each email:

| Panel | Content |
|-------|---------|
| **Rendered View** | HTML email rendered safely in sandboxed iframe |
| **Underlying Content** | Raw text showing actual URLs, hidden text, real link destinations |
| **What Classifier Sees** | Sanitized text fed to the model |

### Security for Rendered View

- Use iframe with `sandbox` attribute (blocks scripts, forms, popups)
- Strip dangerous tags (`<script>`, `<form>`, `<iframe>`)
- Disable all links (replace `href` with `#`)

### Underlying Content Reveals

- Actual URLs (not display text)
- Hidden text (`display:none`, white-on-white, tiny fonts)
- Mismatched link text vs destinations

### File Saving

When user labels an email:
- Generate unique filename: `user_{timestamp}_{hash}.txt`
- Ham → `email-data-set/ham/hard_ham/`
- Spam → `email-data-set/spam/spam/`
- Save cleaned text content to match existing dataset format

### User Flow

1. User visits `/` → sees upload form
2. User uploads `.eml` or `.txt` file → POST to `/upload`
3. Server parses, sanitizes, classifies → returns three-panel result page
4. User reviews content and clicks "Confirm as Spam" or "Confirm as Ham"
5. Server saves file to appropriate dataset folder → redirects to `/` with success message

---

## Part 4: New Dependencies

```toml
[project]
dependencies = [
    # ... existing deps ...
    "flask>=3.0.0",
]

[dependency-groups]
dev = [
    "bandit>=1.9.3",
    "pytest>=8.0.0",
]
```

---

## Part 5: File Structure

New files to create:

```
spam-or-ham-classifier/
├── app.py                    # Flask application
├── templates/
│   ├── index.html            # Upload form
│   └── result.html           # Three-panel review page
├── static/
│   └── style.css             # Styling for panels
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py    # Unit tests for classifier.py
│   ├── test_app.py           # Unit tests for Flask app
│   └── fixtures/
│       ├── sample_ham.txt
│       ├── sample_spam.txt
│       └── sample_html.eml
└── docs/
    └── plans/
        └── 2026-02-03-unit-tests-and-web-app-design.md
```

---

## Implementation Order

All tasks completed:

1. ~~Add pytest to dev dependencies~~ ✅
2. ~~Create test directory structure and fixtures~~ ✅
3. ~~Write unit tests for `classifier.py`~~ ✅ (47 tests)
4. ~~Fix CI workflow (tests + bandit)~~ ✅
5. ~~Add Flask dependency~~ ✅
6. ~~Create `app.py` with routes~~ ✅
7. ~~Create templates (index.html, result.html)~~ ✅
8. ~~Add basic styling~~ ✅ (inline in base.html)
9. ~~Write tests for Flask app~~ ✅ (71 tests)
10. ~~Update README with web app instructions~~ ✅

Total: 118 tests passing
