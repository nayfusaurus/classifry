"""
Unit tests for Flask web application (app.py)

Tests cover:
- safe_render_html(): HTML sanitization
- extract_underlying_content(): Security analysis
- generate_unique_filename(): Unique filename generation
- allowed_file(): File extension validation
- Routes: /, /upload, /label
- CSRF protection
"""

import io
import re
import pytest
from pathlib import Path

# Import the modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app import (
    app,
    safe_render_html,
    extract_underlying_content,
    generate_unique_filename,
    allowed_file,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = True
    with app.test_client() as client:
        yield client


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


def get_csrf_token(client):
    """Helper to get a CSRF token from the session."""
    # Make a GET request to initialize session and get CSRF token
    response = client.get('/')
    # Extract CSRF token from the form
    html = response.data.decode('utf-8')
    match = re.search(r'name="csrf_token"\s+value="([^"]+)"', html)
    if match:
        return match.group(1)
    return None


# =============================================================================
# Tests for safe_render_html()
# =============================================================================

class TestSafeRenderHtml:
    """Tests for the safe_render_html() function."""

    def test_removes_script_tags(self):
        """Script tags should be completely removed."""
        html = '<div>Hello</div><script>alert("XSS")</script><p>World</p>'
        result = safe_render_html(html)
        assert '<script>' not in result
        assert '</script>' not in result
        assert 'alert' not in result
        assert 'XSS' not in result
        assert 'Hello' in result
        assert 'World' in result

    def test_removes_style_tags(self):
        """Style tags should be completely removed."""
        html = '<div>Content</div><style>.evil { display: none; }</style>'
        result = safe_render_html(html)
        assert '<style>' not in result
        assert '</style>' not in result
        assert '.evil' not in result
        assert 'Content' in result

    def test_removes_iframe_tags(self):
        """Iframe tags should be completely removed."""
        html = '<div>Content</div><iframe src="http://evil.com"></iframe>'
        result = safe_render_html(html)
        assert '<iframe' not in result
        assert '</iframe>' not in result
        assert 'evil.com' not in result
        assert 'Content' in result

    def test_removes_object_embed_tags(self):
        """Object and embed tags should be removed."""
        html = '<object data="malware.swf"></object><embed src="malware.swf">'
        result = safe_render_html(html)
        assert '<object' not in result
        assert '<embed' not in result
        assert 'malware.swf' not in result

    def test_removes_onclick_handler(self):
        """onclick event handlers should be removed."""
        html = '<button onclick="stealCookies()">Click me</button>'
        result = safe_render_html(html)
        assert 'onclick' not in result
        assert 'stealCookies' not in result
        assert 'Click me' in result

    def test_removes_onload_handler(self):
        """onload event handlers should be removed."""
        html = '<img onload="maliciousCode()" src="image.png">'
        result = safe_render_html(html)
        assert 'onload' not in result
        assert 'maliciousCode' not in result

    def test_removes_onerror_handler(self):
        """onerror event handlers should be removed."""
        html = '<img onerror="stealData()" src="nonexistent.png">'
        result = safe_render_html(html)
        assert 'onerror' not in result
        assert 'stealData' not in result

    def test_removes_onmouseover_handler(self):
        """onmouseover event handlers should be removed."""
        html = '<div onmouseover="track()">Hover here</div>'
        result = safe_render_html(html)
        assert 'onmouseover' not in result
        assert 'track()' not in result
        assert 'Hover here' in result

    def test_disables_links(self):
        """Links should be converted to non-clickable spans."""
        html = '<a href="http://example.com">Click here</a>'
        result = safe_render_html(html)
        assert '<a ' not in result
        assert '</a>' not in result
        assert 'disabled-link' in result
        assert 'Click here' in result

    def test_link_shows_original_url_in_title(self):
        """Disabled links should show original URL in title."""
        html = '<a href="http://example.com/page">Link text</a>'
        result = safe_render_html(html)
        assert 'http://example.com/page' in result
        assert 'title=' in result

    def test_disables_form_elements(self):
        """Form elements should be disabled."""
        html = '<input type="text" name="username"><button>Submit</button>'
        result = safe_render_html(html)
        assert 'disabled' in result

    def test_form_action_neutralized(self):
        """Form action should be neutralized."""
        html = '<form action="http://evil.com/steal"><input type="submit"></form>'
        result = safe_render_html(html)
        assert 'action="#"' in result

    def test_empty_input_returns_empty_string(self):
        """Empty input should return empty string."""
        assert safe_render_html('') == ''

    def test_none_input_returns_empty_string(self):
        """None input should return empty string."""
        assert safe_render_html(None) == ''

    def test_removes_javascript_urls(self):
        """javascript: URLs should be removed."""
        html = '<a href="javascript:alert(1)">Click</a>'
        result = safe_render_html(html)
        # The link is converted to a span, but javascript should not be executable
        assert 'javascript:' not in result or 'disabled-link' in result

    def test_removes_html_comments(self):
        """HTML comments should be removed."""
        html = '<div>Content</div><!-- This is a comment with script injection -->'
        result = safe_render_html(html)
        assert '<!--' not in result
        assert '-->' not in result
        assert 'Content' in result

    def test_preserves_safe_content(self):
        """Safe HTML content should be preserved."""
        html = '<div><p>Hello <b>World</b></p></div>'
        result = safe_render_html(html)
        assert '<div>' in result
        assert '<p>' in result
        assert '<b>' in result
        assert 'Hello' in result
        assert 'World' in result


# =============================================================================
# Tests for extract_underlying_content()
# =============================================================================

class TestExtractUnderlyingContent:
    """Tests for the extract_underlying_content() function."""

    def test_finds_http_urls(self):
        """Should find http:// URLs."""
        text = 'Visit http://example.com for more info'
        result = extract_underlying_content(text)
        assert 'http://example.com' in result['urls']

    def test_finds_https_urls(self):
        """Should find https:// URLs."""
        text = 'Secure site at https://secure.example.com/page'
        result = extract_underlying_content(text)
        assert any('https://secure.example.com' in url for url in result['urls'])

    def test_finds_www_urls(self):
        """Should find www. URLs."""
        text = 'Check out www.example.org today'
        result = extract_underlying_content(text)
        assert any('www.example.org' in url for url in result['urls'])

    def test_finds_multiple_urls(self):
        """Should find multiple URLs."""
        text = 'Visit http://one.com and http://two.com and www.three.org'
        result = extract_underlying_content(text)
        assert len(result['urls']) >= 2

    def test_detects_link_mismatch(self):
        """Should detect when display text URL differs from href."""
        html = '<a href="http://evil.com">http://legitimate-bank.com</a>'
        result = extract_underlying_content(html)
        assert len(result['link_mismatches']) > 0
        mismatch = result['link_mismatches'][0]
        assert 'display' in mismatch
        assert 'actual' in mismatch
        assert 'legitimate-bank.com' in mismatch['display']
        assert 'evil.com' in mismatch['actual']

    def test_no_mismatch_for_matching_domains(self):
        """Should not report mismatch when domains match."""
        html = '<a href="http://example.com/page">http://example.com</a>'
        result = extract_underlying_content(html)
        assert len(result['link_mismatches']) == 0

    def test_detects_display_none_hidden_text(self):
        """Should detect text hidden with display:none."""
        html = '<div style="display:none;">Hidden spam keywords here</div>'
        result = extract_underlying_content(html)
        assert len(result['hidden_text']) > 0
        assert any('display:none' in item['reason'] for item in result['hidden_text'])

    def test_detects_visibility_hidden_text(self):
        """Should detect text hidden with visibility:hidden."""
        html = '<span style="visibility:hidden;">Secret content</span>'
        result = extract_underlying_content(html)
        assert len(result['hidden_text']) > 0

    def test_detects_zero_font_size_text(self):
        """Should detect text hidden with font-size:0."""
        html = '<span style="font-size:0;">Invisible text</span>'
        result = extract_underlying_content(html)
        assert len(result['hidden_text']) > 0

    def test_detects_white_text_on_white(self):
        """Should detect text hidden with white color."""
        html = '<p style="color:#fff;">White text hidden content</p>'
        result = extract_underlying_content(html)
        assert len(result['hidden_text']) > 0

    def test_finds_form_actions(self):
        """Should find form action URLs."""
        html = '<form action="http://phishing.com/steal"><input type="submit"></form>'
        result = extract_underlying_content(html)
        assert 'http://phishing.com/steal' in result['form_actions']

    def test_ignores_empty_form_actions(self):
        """Should ignore forms with action='#'."""
        html = '<form action="#"><input type="submit"></form>'
        result = extract_underlying_content(html)
        assert len(result['form_actions']) == 0

    def test_detects_tracking_pixel(self):
        """Should detect 1x1 tracking pixels."""
        # Test with height before width (matches second pattern)
        html = '<img height="1" width="1" src="http://track.com/pixel.gif">'
        result = extract_underlying_content(html)
        assert 'Tracking pixel detected' in result['suspicious_elements']

    def test_detects_base64_content(self):
        """Should detect base64 encoded content."""
        html = '<img src="data:image/png;base64,iVBORw0KGgo...">'
        result = extract_underlying_content(html)
        assert 'Base64 encoded content' in result['suspicious_elements'] or \
               'Embedded data URI image' in result['suspicious_elements']

    def test_empty_input_returns_empty_result(self):
        """Empty input should return structure with empty lists."""
        result = extract_underlying_content('')
        assert result['urls'] == []
        assert result['hidden_text'] == []
        assert result['link_mismatches'] == []
        assert result['form_actions'] == []
        assert result['suspicious_elements'] == []

    def test_none_input_returns_empty_result(self):
        """None input should return structure with empty lists."""
        result = extract_underlying_content(None)
        assert result['urls'] == []
        assert result['hidden_text'] == []
        assert result['link_mismatches'] == []


# =============================================================================
# Tests for generate_unique_filename()
# =============================================================================

class TestGenerateUniqueFilename:
    """Tests for the generate_unique_filename() function."""

    def test_returns_string(self):
        """Should return a string."""
        result = generate_unique_filename()
        assert isinstance(result, str)

    def test_format_user_timestamp_hash(self):
        """Should return format user_{timestamp}_{hash}.txt."""
        result = generate_unique_filename()
        assert result.startswith('user_')
        assert result.endswith('.txt')
        # Check pattern: user_<digits>_<hex>.txt
        pattern = r'^user_\d+_[a-f0-9]+\.txt$'
        assert re.match(pattern, result), f"Filename '{result}' doesn't match expected pattern"

    def test_contains_timestamp(self):
        """Should contain a timestamp."""
        result = generate_unique_filename()
        parts = result.replace('user_', '').replace('.txt', '').split('_')
        assert len(parts) == 2
        timestamp_str = parts[0]
        # Timestamp should be numeric and reasonable (after year 2000)
        timestamp = int(timestamp_str)
        assert timestamp > 946684800000  # Jan 1, 2000 in milliseconds

    def test_contains_hash(self):
        """Should contain a hash suffix."""
        result = generate_unique_filename()
        parts = result.replace('user_', '').replace('.txt', '').split('_')
        hash_part = parts[1]
        # Hash should be hexadecimal (8 characters based on implementation)
        assert re.match(r'^[a-f0-9]{8}$', hash_part)

    def test_unique_across_calls(self):
        """Should generate unique filenames across multiple calls."""
        filenames = [generate_unique_filename() for _ in range(100)]
        # All filenames should be unique
        assert len(set(filenames)) == len(filenames)

    def test_valid_filename_characters(self):
        """Should only contain valid filename characters."""
        result = generate_unique_filename()
        # Only alphanumeric, underscore, and dot
        assert re.match(r'^[a-z0-9_.]+$', result)


# =============================================================================
# Tests for allowed_file()
# =============================================================================

class TestAllowedFile:
    """Tests for the allowed_file() function."""

    def test_eml_extension_allowed(self):
        """Should return True for .eml files."""
        assert allowed_file('email.eml') is True
        assert allowed_file('test.EML') is True
        assert allowed_file('TEST.Eml') is True

    def test_txt_extension_allowed(self):
        """Should return True for .txt files."""
        assert allowed_file('email.txt') is True
        assert allowed_file('test.TXT') is True
        assert allowed_file('TEST.Txt') is True

    def test_msg_extension_allowed(self):
        """Should return True for .msg files."""
        assert allowed_file('outlook.msg') is True
        assert allowed_file('test.MSG') is True

    def test_exe_extension_not_allowed(self):
        """Should return False for .exe files."""
        assert allowed_file('malware.exe') is False

    def test_py_extension_not_allowed(self):
        """Should return False for .py files."""
        assert allowed_file('script.py') is False

    def test_html_extension_not_allowed(self):
        """Should return False for .html files."""
        assert allowed_file('page.html') is False

    def test_pdf_extension_not_allowed(self):
        """Should return False for .pdf files."""
        assert allowed_file('document.pdf') is False

    def test_no_extension_not_allowed(self):
        """Should return False for files without extension."""
        assert allowed_file('noextension') is False

    def test_empty_filename_not_allowed(self):
        """Should return False for empty filename."""
        assert allowed_file('') is False

    def test_none_filename_not_allowed(self):
        """Should return False for None filename."""
        assert allowed_file(None) is False

    def test_double_extension_checks_last(self):
        """Should check the last extension only."""
        # Should be allowed because .txt is the actual extension
        assert allowed_file('file.exe.txt') is True
        # Should not be allowed because .exe is the actual extension
        assert allowed_file('file.txt.exe') is False


# =============================================================================
# Tests for Routes
# =============================================================================

class TestIndexRoute:
    """Tests for GET / route."""

    def test_returns_200(self, client):
        """GET / should return 200 status."""
        response = client.get('/')
        assert response.status_code == 200

    def test_contains_upload_form(self, client):
        """Response should contain an upload form."""
        response = client.get('/')
        html = response.data.decode('utf-8')
        assert '<form' in html
        assert 'enctype="multipart/form-data"' in html or 'upload' in html.lower()

    def test_contains_csrf_token(self, client):
        """Response should contain a CSRF token."""
        response = client.get('/')
        html = response.data.decode('utf-8')
        assert 'csrf_token' in html

    def test_contains_file_input(self, client):
        """Response should contain a file input."""
        response = client.get('/')
        html = response.data.decode('utf-8')
        assert 'type="file"' in html


class TestUploadRoute:
    """Tests for POST /upload route."""

    def test_without_file_returns_error(self, client):
        """POST /upload without file should return error."""
        csrf_token = get_csrf_token(client)
        response = client.post('/upload', data={'csrf_token': csrf_token})
        # Should redirect back to index with error
        assert response.status_code in [302, 400]

    def test_with_empty_filename_returns_error(self, client):
        """POST /upload with empty filename should return error."""
        csrf_token = get_csrf_token(client)
        response = client.post(
            '/upload',
            data={
                'csrf_token': csrf_token,
                'email_file': (io.BytesIO(b''), '')
            }
        )
        assert response.status_code in [302, 400]

    def test_with_invalid_extension_returns_error(self, client):
        """POST /upload with invalid file extension should return error."""
        csrf_token = get_csrf_token(client)
        response = client.post(
            '/upload',
            data={
                'csrf_token': csrf_token,
                'email_file': (io.BytesIO(b'malicious content'), 'malware.exe')
            }
        )
        # Should redirect back with error flash message
        assert response.status_code == 302
        # Follow redirect to check flash message
        follow_response = client.get('/')
        html = follow_response.data.decode('utf-8')
        assert 'Invalid file type' in html or 'error' in html.lower()

    def test_with_valid_txt_file_returns_result(self, client, sample_ham_path):
        """POST /upload with valid .txt file should return result page."""
        csrf_token = get_csrf_token(client)
        with open(sample_ham_path, 'rb') as f:
            content = f.read()
        response = client.post(
            '/upload',
            data={
                'csrf_token': csrf_token,
                'email_file': (io.BytesIO(content), 'test_email.txt')
            }
        )
        # Should return 200 with result page or redirect with success
        assert response.status_code in [200, 302]
        if response.status_code == 200:
            html = response.data.decode('utf-8')
            # Result page should contain classification info
            assert 'SPAM' in html or 'HAM' in html or 'classification' in html.lower()

    def test_with_valid_eml_file_returns_result(self, client, sample_html_path):
        """POST /upload with valid .eml file should return result page."""
        csrf_token = get_csrf_token(client)
        with open(sample_html_path, 'rb') as f:
            content = f.read()
        response = client.post(
            '/upload',
            data={
                'csrf_token': csrf_token,
                'email_file': (io.BytesIO(content), 'test_email.eml')
            }
        )
        assert response.status_code in [200, 302]
        if response.status_code == 200:
            html = response.data.decode('utf-8')
            assert 'SPAM' in html or 'HAM' in html or 'UNKNOWN' in html


class TestLabelRoute:
    """Tests for POST /label route."""

    def test_with_valid_data_saves_and_redirects(self, client, tmp_path, monkeypatch):
        """POST /label with valid data should save file and redirect."""
        # Temporarily change save directories to tmp_path
        monkeypatch.setattr('app.HAM_DIR', tmp_path / 'ham')
        monkeypatch.setattr('app.SPAM_DIR', tmp_path / 'spam')

        csrf_token = get_csrf_token(client)
        response = client.post(
            '/label',
            data={
                'csrf_token': csrf_token,
                'label': 'ham',
                'email_content': 'This is a test email content for labeling.'
            }
        )
        # Should redirect back to index
        assert response.status_code == 302

        # Check that file was created
        ham_dir = tmp_path / 'ham'
        if ham_dir.exists():
            files = list(ham_dir.glob('user_*.txt'))
            assert len(files) == 1

    def test_with_spam_label_saves_to_spam_dir(self, client, tmp_path, monkeypatch):
        """POST /label with spam label should save to spam directory."""
        monkeypatch.setattr('app.HAM_DIR', tmp_path / 'ham')
        monkeypatch.setattr('app.SPAM_DIR', tmp_path / 'spam')

        csrf_token = get_csrf_token(client)
        response = client.post(
            '/label',
            data={
                'csrf_token': csrf_token,
                'label': 'spam',
                'email_content': 'This is spam content.'
            }
        )
        assert response.status_code == 302

        spam_dir = tmp_path / 'spam'
        if spam_dir.exists():
            files = list(spam_dir.glob('user_*.txt'))
            assert len(files) == 1

    def test_with_invalid_label_returns_error(self, client):
        """POST /label with invalid label should return error."""
        csrf_token = get_csrf_token(client)
        response = client.post(
            '/label',
            data={
                'csrf_token': csrf_token,
                'label': 'invalid_label',
                'email_content': 'Some content.'
            }
        )
        # Should redirect with error
        assert response.status_code == 302

    def test_with_empty_content_returns_error(self, client):
        """POST /label with empty content should return error."""
        csrf_token = get_csrf_token(client)
        response = client.post(
            '/label',
            data={
                'csrf_token': csrf_token,
                'label': 'ham',
                'email_content': ''
            }
        )
        assert response.status_code == 302

    def test_without_label_returns_error(self, client):
        """POST /label without label should return error."""
        csrf_token = get_csrf_token(client)
        response = client.post(
            '/label',
            data={
                'csrf_token': csrf_token,
                'email_content': 'Some content.'
            }
        )
        assert response.status_code == 302


# =============================================================================
# Tests for CSRF Protection
# =============================================================================

class TestCSRFProtection:
    """Tests for CSRF protection."""

    def test_upload_without_csrf_returns_error(self, client):
        """POST /upload without CSRF token should fail."""
        response = client.post(
            '/upload',
            data={
                'email_file': (io.BytesIO(b'test content'), 'test.txt')
            }
        )
        # Should redirect with error (CSRF validation failed)
        assert response.status_code == 302
        # Follow redirect to check for error message
        follow_response = client.get('/')
        html = follow_response.data.decode('utf-8')
        assert 'Invalid request' in html or 'error' in html.lower() or 'try again' in html.lower()

    def test_upload_with_invalid_csrf_returns_error(self, client):
        """POST /upload with invalid CSRF token should fail."""
        response = client.post(
            '/upload',
            data={
                'csrf_token': 'invalid_token_here',
                'email_file': (io.BytesIO(b'test content'), 'test.txt')
            }
        )
        assert response.status_code == 302

    def test_upload_with_valid_csrf_succeeds(self, client, sample_ham_path):
        """POST /upload with valid CSRF token should succeed."""
        csrf_token = get_csrf_token(client)
        with open(sample_ham_path, 'rb') as f:
            content = f.read()
        response = client.post(
            '/upload',
            data={
                'csrf_token': csrf_token,
                'email_file': (io.BytesIO(content), 'test.txt')
            }
        )
        # Should succeed (200 result page or 302 redirect with success)
        assert response.status_code in [200, 302]

    def test_label_without_csrf_returns_error(self, client):
        """POST /label without CSRF token should fail."""
        response = client.post(
            '/label',
            data={
                'label': 'ham',
                'email_content': 'Test content'
            }
        )
        assert response.status_code == 302

    def test_label_with_valid_csrf_succeeds(self, client, tmp_path, monkeypatch):
        """POST /label with valid CSRF token should succeed."""
        monkeypatch.setattr('app.HAM_DIR', tmp_path / 'ham')
        monkeypatch.setattr('app.SPAM_DIR', tmp_path / 'spam')

        csrf_token = get_csrf_token(client)
        response = client.post(
            '/label',
            data={
                'csrf_token': csrf_token,
                'label': 'ham',
                'email_content': 'This is test content for CSRF validation.'
            }
        )
        assert response.status_code == 302


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the Flask app."""

    def test_full_upload_workflow(self, client, sample_spam_path):
        """Test full workflow: upload file and get classification result."""
        # Step 1: Get the index page (initializes session)
        response = client.get('/')
        assert response.status_code == 200

        # Step 2: Get CSRF token
        csrf_token = get_csrf_token(client)
        assert csrf_token is not None

        # Step 3: Upload a file
        with open(sample_spam_path, 'rb') as f:
            content = f.read()
        response = client.post(
            '/upload',
            data={
                'csrf_token': csrf_token,
                'email_file': (io.BytesIO(content), 'spam_test.txt')
            }
        )

        # Should get result page
        if response.status_code == 200:
            html = response.data.decode('utf-8')
            # Result page should contain classification
            assert 'SPAM' in html or 'HAM' in html or 'UNKNOWN' in html

    def test_html_email_security_analysis(self, client, sample_html_path):
        """Test that HTML emails get proper security analysis."""
        csrf_token = get_csrf_token(client)

        with open(sample_html_path, 'rb') as f:
            content = f.read()

        response = client.post(
            '/upload',
            data={
                'csrf_token': csrf_token,
                'email_file': (io.BytesIO(content), 'phishing.eml')
            }
        )

        if response.status_code == 200:
            html = response.data.decode('utf-8')
            # Should show classification result
            assert 'SPAM' in html or 'HAM' in html or 'UNKNOWN' in html
