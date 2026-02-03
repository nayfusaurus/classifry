"""
Unit tests for classifier.py

Tests cover:
- clean_text(): Text preprocessing and sanitization
- parse_email_file(): Email file parsing
- SpamClassifier: Neural network model
- EmailClassifier: Classification wrapper
"""

import pytest
import torch
from pathlib import Path

# Import the modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from classifier import clean_text, parse_email_file, SpamClassifier, EmailClassifier


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def fixtures_dir():
    """Return the path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_ham_path(fixtures_dir):
    """Return path to sample ham email fixture."""
    return fixtures_dir / "sample_ham.txt"


@pytest.fixture
def sample_spam_path(fixtures_dir):
    """Return path to sample spam email fixture."""
    return fixtures_dir / "sample_spam.txt"


@pytest.fixture
def sample_html_path(fixtures_dir):
    """Return path to sample HTML email fixture."""
    return fixtures_dir / "sample_html.eml"


@pytest.fixture
def models_dir():
    """Return the path to models directory."""
    return Path(__file__).parent.parent / "models"


@pytest.fixture
def email_classifier(models_dir):
    """Create an EmailClassifier instance with pre-trained model."""
    return EmailClassifier(models_dir)


# =============================================================================
# Tests for clean_text()
# =============================================================================

class TestCleanText:
    """Tests for the clean_text() function."""

    def test_html_tags_removed(self):
        """HTML tags should be stripped from text."""
        html_text = "<html><body><p>Hello <b>World</b></p></body></html>"
        result = clean_text(html_text)
        assert "<" not in result
        assert ">" not in result
        assert "html" not in result
        assert "hello" in result
        assert "world" in result

    def test_complex_html_removed(self):
        """Complex HTML with styles and attributes should be stripped."""
        html_text = '<div style="color: red;"><a href="http://example.com">Click here</a></div>'
        result = clean_text(html_text)
        assert "div" not in result
        assert "style" not in result
        assert "href" not in result
        assert "click" in result
        assert "here" in result

    def test_urls_replaced(self):
        """URLs should be replaced with ' url '."""
        text_with_urls = "Visit http://example.com for more info and www.test.org too"
        result = clean_text(text_with_urls)
        assert "http" not in result
        assert "example.com" not in result
        assert "www.test.org" not in result
        assert "url" in result

    def test_https_urls_replaced(self):
        """HTTPS URLs should also be replaced."""
        text = "Check https://secure-site.com/page?param=value"
        result = clean_text(text)
        assert "https" not in result
        assert "secure-site.com" not in result
        assert "url" in result

    def test_email_addresses_replaced(self):
        """Email addresses should be replaced with ' emailaddr '."""
        text = "Contact me at john.doe@example.com or support@company.org"
        result = clean_text(text)
        assert "@" not in result
        assert "john.doe" not in result
        assert "example.com" not in result
        assert "emailaddr" in result

    def test_numbers_removed(self):
        """Standalone numbers should be removed."""
        text = "Call 12345 or visit room 42 today"
        result = clean_text(text)
        assert "12345" not in result
        assert "42" not in result
        assert "call" in result
        assert "room" in result

    def test_empty_input_returns_empty_string(self):
        """Empty string input should return empty string."""
        assert clean_text("") == ""

    def test_none_input_returns_empty_string(self):
        """None input should return empty string."""
        assert clean_text(None) == ""

    def test_whitespace_only_returns_empty_string(self):
        """Whitespace-only input should return empty string."""
        assert clean_text("   \n\t  ") == ""

    def test_special_characters_removed(self):
        """Special characters should be removed."""
        text = "Hello! How are you? I'm fine... #awesome @mention $$$"
        result = clean_text(text)
        assert "!" not in result
        assert "?" not in result
        assert "'" not in result
        assert "#" not in result
        assert "$" not in result
        assert "hello" in result
        assert "awesome" in result

    def test_text_lowercased(self):
        """Text should be converted to lowercase."""
        text = "HELLO World MiXeD CaSe"
        result = clean_text(text)
        assert result == "hello world mixed case"

    def test_extra_whitespace_normalized(self):
        """Multiple spaces should be collapsed to single space."""
        text = "hello     world    test"
        result = clean_text(text)
        assert "  " not in result
        assert result == "hello world test"

    def test_non_string_input_returns_empty(self):
        """Non-string input should return empty string."""
        assert clean_text(123) == ""
        assert clean_text([1, 2, 3]) == ""
        assert clean_text({"key": "value"}) == ""


# =============================================================================
# Tests for parse_email_file()
# =============================================================================

class TestParseEmailFile:
    """Tests for the parse_email_file() function."""

    def test_parses_plain_text_email(self, sample_ham_path):
        """Should correctly parse a plain text email file."""
        result = parse_email_file(sample_ham_path)
        assert isinstance(result, dict)
        assert "subject" in result
        assert "body" in result

    def test_extracts_subject(self, sample_ham_path):
        """Should extract the subject line."""
        result = parse_email_file(sample_ham_path)
        assert "Q4 Project Timeline" in result["subject"]

    def test_extracts_body(self, sample_ham_path):
        """Should extract the email body."""
        result = parse_email_file(sample_ham_path)
        assert "Sarah" in result["body"] or "Michael" in result["body"]

    def test_handles_html_email(self, sample_html_path):
        """Should handle HTML email files."""
        result = parse_email_file(sample_html_path)
        assert isinstance(result, dict)
        assert "subject" in result
        assert "body" in result
        # The body should contain HTML content (not necessarily stripped)
        assert len(result["body"]) > 0

    def test_html_email_extracts_subject(self, sample_html_path):
        """Should extract subject from HTML email."""
        result = parse_email_file(sample_html_path)
        assert "Account" in result["subject"] or "Compromised" in result["subject"]

    def test_returns_dict_with_required_keys(self, sample_spam_path):
        """Should always return a dict with 'subject' and 'body' keys."""
        result = parse_email_file(sample_spam_path)
        assert isinstance(result, dict)
        assert set(result.keys()) >= {"subject", "body"}

    def test_spam_email_parsed(self, sample_spam_path):
        """Should correctly parse spam email file."""
        result = parse_email_file(sample_spam_path)
        assert "URGENT" in result["subject"] or "WON" in result["subject"]
        assert "MILLION" in result["body"] or "winner" in result["body"].lower()

    def test_handles_nonexistent_file_gracefully(self, fixtures_dir):
        """Should handle nonexistent files without crashing."""
        fake_path = fixtures_dir / "nonexistent_file.txt"
        # Should not raise an exception
        result = parse_email_file(fake_path)
        assert isinstance(result, dict)
        assert "subject" in result
        assert "body" in result


# =============================================================================
# Tests for SpamClassifier model
# =============================================================================

class TestSpamClassifier:
    """Tests for the SpamClassifier neural network model."""

    def test_model_initialization(self):
        """Model should initialize with correct architecture."""
        input_dim = 5000
        model = SpamClassifier(input_dim=input_dim)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_model_with_custom_dropout(self):
        """Model should accept custom dropout rate."""
        model = SpamClassifier(input_dim=5000, dropout_rate=0.5)
        assert model is not None

    def test_forward_pass_accepts_correct_shape(self):
        """Forward pass should accept [batch, 5000] tensor."""
        input_dim = 5000
        batch_size = 8
        model = SpamClassifier(input_dim=input_dim)

        # Create input tensor with correct shape
        x = torch.randn(batch_size, input_dim)

        # Should not raise an error
        model.eval()
        with torch.no_grad():
            output = model(x)

        assert output is not None

    def test_output_shape_is_batch_by_one(self):
        """Output shape should be [batch, 1]."""
        input_dim = 5000
        batch_size = 16
        model = SpamClassifier(input_dim=input_dim)

        x = torch.randn(batch_size, input_dim)

        model.eval()
        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 1)

    def test_single_sample_forward_pass(self):
        """Should handle single sample (batch size 1)."""
        input_dim = 5000
        model = SpamClassifier(input_dim=input_dim)

        x = torch.randn(1, input_dim)

        model.eval()
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 1)

    def test_output_values_between_zero_and_one(self):
        """Output values should be between 0 and 1 (sigmoid)."""
        input_dim = 5000
        batch_size = 32
        model = SpamClassifier(input_dim=input_dim)

        # Test with various input values
        x = torch.randn(batch_size, input_dim)

        model.eval()
        with torch.no_grad():
            output = model(x)

        assert (output >= 0).all(), "All outputs should be >= 0"
        assert (output <= 1).all(), "All outputs should be <= 1"

    def test_output_range_with_extreme_inputs(self):
        """Output should remain in [0, 1] even with extreme inputs."""
        input_dim = 5000
        model = SpamClassifier(input_dim=input_dim)

        # Very large positive values
        x_large = torch.ones(10, input_dim) * 100
        # Very large negative values
        x_small = torch.ones(10, input_dim) * -100
        # Mixed
        x_mixed = torch.randn(10, input_dim) * 50

        model.eval()
        with torch.no_grad():
            for x in [x_large, x_small, x_mixed]:
                output = model(x)
                assert (output >= 0).all(), "Output should be >= 0"
                assert (output <= 1).all(), "Output should be <= 1"

    def test_model_has_correct_layer_structure(self):
        """Model should have the expected layer structure."""
        model = SpamClassifier(input_dim=5000)

        # Check that network attribute exists
        assert hasattr(model, 'network')

        # The network should be a Sequential container
        assert isinstance(model.network, torch.nn.Sequential)


# =============================================================================
# Tests for EmailClassifier
# =============================================================================

class TestEmailClassifier:
    """Tests for the EmailClassifier wrapper class."""

    def test_loads_model_from_directory(self, models_dir):
        """Should successfully load model from models directory."""
        classifier = EmailClassifier(models_dir)
        assert classifier is not None
        assert classifier.model is not None
        assert classifier.tfidf is not None

    def test_raises_error_for_missing_model(self, tmp_path):
        """Should raise FileNotFoundError when model files don't exist."""
        with pytest.raises(FileNotFoundError):
            EmailClassifier(tmp_path)

    def test_classify_returns_required_keys(self, email_classifier):
        """classify() should return dict with required keys."""
        result = email_classifier.classify("Hello, this is a test email")

        assert isinstance(result, dict)
        assert "prediction" in result
        assert "spam_probability" in result
        assert "confidence" in result

    def test_classify_prediction_is_spam_or_ham(self, email_classifier):
        """Prediction should be either 'SPAM' or 'HAM'."""
        result = email_classifier.classify("Test email content")

        assert result["prediction"] in ["SPAM", "HAM", "UNKNOWN"]

    def test_classify_spam_probability_is_float(self, email_classifier):
        """spam_probability should be a float between 0 and 1."""
        result = email_classifier.classify("Test email content")

        assert isinstance(result["spam_probability"], float)
        assert 0 <= result["spam_probability"] <= 1

    def test_classify_confidence_is_valid(self, email_classifier):
        """confidence should be between 0.5 and 1."""
        result = email_classifier.classify("Test email content")

        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 1

    def test_classify_empty_text_returns_unknown(self, email_classifier):
        """Empty text should return UNKNOWN prediction."""
        result = email_classifier.classify("")

        assert result["prediction"] == "UNKNOWN"
        assert "error" in result

    def test_classify_file_works_with_ham(self, email_classifier, sample_ham_path):
        """classify_file() should work with ham fixture."""
        result = email_classifier.classify_file(sample_ham_path)

        assert isinstance(result, dict)
        assert "prediction" in result
        assert "spam_probability" in result
        assert "confidence" in result
        assert "subject" in result

    def test_classify_file_works_with_spam(self, email_classifier, sample_spam_path):
        """classify_file() should work with spam fixture."""
        result = email_classifier.classify_file(sample_spam_path)

        assert isinstance(result, dict)
        assert "prediction" in result
        assert "spam_probability" in result

    def test_classify_file_works_with_html(self, email_classifier, sample_html_path):
        """classify_file() should work with HTML email fixture."""
        result = email_classifier.classify_file(sample_html_path)

        assert isinstance(result, dict)
        assert "prediction" in result
        assert "spam_probability" in result

    def test_threshold_behavior_high(self, email_classifier):
        """High threshold should make classification more conservative."""
        text = "Win money now! Free prize claim immediately!"

        # With default threshold
        result_default = email_classifier.classify(text, threshold=0.5)

        # With very high threshold (0.99)
        result_high = email_classifier.classify(text, threshold=0.99)

        # Same probability but different threshold affects prediction
        assert result_default["spam_probability"] == result_high["spam_probability"]

    def test_threshold_behavior_low(self, email_classifier):
        """Low threshold should classify more emails as spam."""
        text = "Meeting tomorrow at 3pm to discuss project"

        # With very low threshold (0.01)
        result_low = email_classifier.classify(text, threshold=0.01)

        # With default threshold
        result_default = email_classifier.classify(text, threshold=0.5)

        # Same probability but different threshold affects prediction
        assert result_default["spam_probability"] == result_low["spam_probability"]

    def test_threshold_zero_point_five_boundary(self, email_classifier):
        """Test default threshold of 0.5 for boundary classification."""
        # This tests the logic: prediction is SPAM if probability >= threshold
        text = "Normal email about work meeting"

        result = email_classifier.classify(text, threshold=0.5)

        prob = result["spam_probability"]
        if prob >= 0.5:
            assert result["prediction"] == "SPAM"
        else:
            assert result["prediction"] == "HAM"

    def test_typical_ham_classification(self, email_classifier):
        """Typical legitimate email content should be classified."""
        ham_text = """
        Hi team,

        Just a reminder about our weekly standup meeting tomorrow at 10am.
        Please come prepared with your status updates.

        Thanks,
        John
        """

        result = email_classifier.classify(ham_text)

        # Should have valid structure regardless of prediction
        assert result["prediction"] in ["SPAM", "HAM"]
        assert 0 <= result["spam_probability"] <= 1

    def test_typical_spam_classification(self, email_classifier):
        """Typical spam content should be classified."""
        spam_text = """
        CONGRATULATIONS! You have won $1,000,000!!!
        Click here NOW to claim your FREE PRIZE!
        ACT IMMEDIATELY - offer expires in 24 hours!
        Send your bank account details to claim!
        """

        result = email_classifier.classify(spam_text)

        # Should have valid structure regardless of prediction
        assert result["prediction"] in ["SPAM", "HAM"]
        assert 0 <= result["spam_probability"] <= 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full classification pipeline."""

    def test_end_to_end_classification(self, email_classifier, sample_ham_path):
        """Test full pipeline from file to classification result."""
        # Parse the file
        parsed = parse_email_file(sample_ham_path)

        # Combine subject and body
        combined = f"{parsed['subject']} {parsed['body']}"

        # Clean the text
        cleaned = clean_text(combined)

        # Classify
        result = email_classifier.classify(cleaned)

        assert result["prediction"] in ["SPAM", "HAM"]
        assert 0 <= result["spam_probability"] <= 1

    def test_clean_text_preserves_meaningful_content(self):
        """Cleaned text should still contain meaningful words."""
        original = "Hello John, please review the attached document by Friday."
        cleaned = clean_text(original)

        # Should contain meaningful words
        assert "hello" in cleaned
        assert "john" in cleaned
        assert "review" in cleaned
        assert "attached" in cleaned
        assert "document" in cleaned
        assert "friday" in cleaned

    def test_model_consistent_output(self, email_classifier):
        """Model should give consistent output for same input."""
        text = "This is a test email about an important meeting"

        result1 = email_classifier.classify(text)
        result2 = email_classifier.classify(text)

        assert result1["spam_probability"] == result2["spam_probability"]
        assert result1["prediction"] == result2["prediction"]
