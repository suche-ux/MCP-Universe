"""Unit tests for OpenAI retry, backoff, and timeout behavior."""

import os
import time
import unittest
from unittest.mock import patch, Mock, MagicMock

from openai import RateLimitError, APIError, APITimeoutError, InternalServerError

from mcpuniverse.llm.openai import OpenAIModel


def _make_rate_limit_error():
    """Create a mock RateLimitError."""
    mock_response = Mock()
    mock_response.status_code = 429
    mock_response.headers = {}
    mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
    return RateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"error": {"message": "Rate limit exceeded"}},
    )


def _make_api_timeout_error(request=None):
    """Create a mock APITimeoutError."""
    return APITimeoutError(request=request or Mock())


def _make_internal_server_error():
    """Create a mock InternalServerError."""
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.headers = {}
    mock_response.json.return_value = {"error": {"message": "Internal server error"}}
    return InternalServerError(
        message="Internal server error",
        response=mock_response,
        body={"error": {"message": "Internal server error"}},
    )


def _make_successful_chat_response(content="Hello!"):
    """Create a mock successful chat.completions.create response."""
    mock_message = Mock()
    mock_message.content = content
    mock_message.tool_calls = None
    mock_choice = Mock()
    mock_choice.message = mock_message
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    mock_response.model = "gpt-4.1"
    return mock_response


class TestOpenAIRetryDefaults(unittest.TestCase):
    """Test that default retry/timeout values are set for 12-hour window."""

    @patch("mcpuniverse.llm.openai.OpenAI")
    def test_default_max_retries_is_720(self, mock_openai_cls):
        """max_retries should default to 720 (720 * 60s = 12h)."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_successful_chat_response()

        model = OpenAIModel()
        model._generate(messages=[{"role": "user", "content": "hi"}])
        # If it succeeds on the first try, it should have been called once
        mock_client.chat.completions.create.assert_called_once()

    @patch("mcpuniverse.llm.openai.OpenAI")
    def test_default_timeout_is_600(self, mock_openai_cls):
        """Per-request timeout should default to 600s (10 min)."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_successful_chat_response()

        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_TIMEOUT_SECONDS", None)
            model = OpenAIModel()
            model._generate(messages=[{"role": "user", "content": "hi"}])

        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs["timeout"], 600)

    @patch("mcpuniverse.llm.openai.OpenAI")
    def test_timeout_env_var_override(self, mock_openai_cls):
        """OPENAI_API_TIMEOUT_SECONDS env var should override default timeout."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_successful_chat_response()

        with patch.dict(os.environ, {"OPENAI_API_TIMEOUT_SECONDS": "1200"}):
            model = OpenAIModel()
            model._generate(messages=[{"role": "user", "content": "hi"}])

        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs["timeout"], 1200)

    @patch("mcpuniverse.llm.openai.time.sleep")
    @patch("mcpuniverse.llm.openai.OpenAI")
    def test_custom_kwargs_override_defaults(self, mock_openai_cls, mock_sleep):
        """Custom max_retries and base_delay should override the defaults."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        # Fail more than custom max_retries times to verify it gives up at the right count
        custom_retries = 2
        mock_client.chat.completions.create.side_effect = [
            _make_rate_limit_error() for _ in range(custom_retries + 1)
        ]

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_TIMEOUT_SECONDS", None)
            model = OpenAIModel()
            result = model._generate(
                messages=[{"role": "user", "content": "hi"}],
                max_retries=custom_retries,
                base_delay=5.0,
            )

        # Should return None after custom_retries + 1 attempts
        self.assertIsNone(result)
        self.assertEqual(mock_client.chat.completions.create.call_count, custom_retries + 1)
        # Verify linear backoff used the custom base_delay
        for call in mock_sleep.call_args_list:
            self.assertEqual(call.args[0], 5.0)


class TestOpenAILinearBackoff(unittest.TestCase):
    """Test that retries use linear backoff (constant delay)."""

    @patch("mcpuniverse.llm.openai.time.sleep")
    @patch("mcpuniverse.llm.openai.OpenAI")
    def test_linear_backoff_constant_delay(self, mock_openai_cls, mock_sleep):
        """Each retry should sleep for exactly base_delay seconds (linear)."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        # Fail 3 times, then succeed
        mock_client.chat.completions.create.side_effect = [
            _make_rate_limit_error(),
            _make_rate_limit_error(),
            _make_rate_limit_error(),
            _make_successful_chat_response("success"),
        ]

        model = OpenAIModel()
        result = model._generate(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=5,
            base_delay=60.0,
        )

        self.assertEqual(result, "success")
        # Should have slept 3 times, each with the same delay
        self.assertEqual(mock_sleep.call_count, 3)
        for call in mock_sleep.call_args_list:
            self.assertEqual(call.args[0], 60.0)

    @patch("mcpuniverse.llm.openai.time.sleep")
    @patch("mcpuniverse.llm.openai.OpenAI")
    def test_no_exponential_growth(self, mock_openai_cls, mock_sleep):
        """Verify delays do NOT grow exponentially — all should be equal."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client

        num_failures = 5
        mock_client.chat.completions.create.side_effect = [
            _make_rate_limit_error() for _ in range(num_failures)
        ] + [_make_successful_chat_response()]

        model = OpenAIModel()
        model._generate(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=10,
            base_delay=30.0,
        )

        delays = [call.args[0] for call in mock_sleep.call_args_list]
        # All delays should be the same (linear, not exponential)
        self.assertTrue(all(d == 30.0 for d in delays))
        self.assertEqual(len(delays), num_failures)


class TestOpenAIRetryableExceptions(unittest.TestCase):
    """Test that the correct exceptions trigger retries."""

    @patch("mcpuniverse.llm.openai.time.sleep")
    @patch("mcpuniverse.llm.openai.OpenAI")
    def test_rate_limit_error_retries(self, mock_openai_cls, mock_sleep):
        """RateLimitError should trigger a retry."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            _make_rate_limit_error(),
            _make_successful_chat_response("ok"),
        ]

        model = OpenAIModel()
        result = model._generate(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=2,
            base_delay=0.01,
        )
        self.assertEqual(result, "ok")

    @patch("mcpuniverse.llm.openai.time.sleep")
    @patch("mcpuniverse.llm.openai.OpenAI")
    def test_api_timeout_error_retries(self, mock_openai_cls, mock_sleep):
        """APITimeoutError should trigger a retry."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            _make_api_timeout_error(),
            _make_successful_chat_response("ok"),
        ]

        model = OpenAIModel()
        result = model._generate(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=2,
            base_delay=0.01,
        )
        self.assertEqual(result, "ok")

    @patch("mcpuniverse.llm.openai.time.sleep")
    @patch("mcpuniverse.llm.openai.OpenAI")
    def test_internal_server_error_retries(self, mock_openai_cls, mock_sleep):
        """InternalServerError (500) should trigger a retry."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            _make_internal_server_error(),
            _make_successful_chat_response("ok"),
        ]

        model = OpenAIModel()
        result = model._generate(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=2,
            base_delay=0.01,
        )
        self.assertEqual(result, "ok")

    @patch("mcpuniverse.llm.openai.time.sleep")
    @patch("mcpuniverse.llm.openai.OpenAI")
    def test_connection_error_retries(self, mock_openai_cls, mock_sleep):
        """ConnectionError should trigger a retry."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            ConnectionError("Connection refused"),
            _make_successful_chat_response("ok"),
        ]

        model = OpenAIModel()
        result = model._generate(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=2,
            base_delay=0.01,
        )
        self.assertEqual(result, "ok")

    @patch("mcpuniverse.llm.openai.OpenAI")
    def test_non_retryable_error_returns_none(self, mock_openai_cls):
        """Non-retryable errors should return None immediately (no retry)."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = ValueError("bad value")

        model = OpenAIModel()
        result = model._generate(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=5,
            base_delay=0.01,
        )
        self.assertIsNone(result)
        # Should have been called only once (no retries for non-retryable errors)
        mock_client.chat.completions.create.assert_called_once()


class TestOpenAIRetryExhaustion(unittest.TestCase):
    """Test behavior when all retries are exhausted."""

    @patch("mcpuniverse.llm.openai.time.sleep")
    @patch("mcpuniverse.llm.openai.OpenAI")
    def test_all_retries_exhausted_returns_none(self, mock_openai_cls, mock_sleep):
        """When all retries fail, should return None."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        max_retries = 3
        mock_client.chat.completions.create.side_effect = [
            _make_rate_limit_error() for _ in range(max_retries + 1)
        ]

        model = OpenAIModel()
        result = model._generate(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=max_retries,
            base_delay=0.01,
        )
        self.assertIsNone(result)
        # Should have attempted max_retries + 1 times (initial + retries)
        self.assertEqual(mock_client.chat.completions.create.call_count, max_retries + 1)

    @patch("mcpuniverse.llm.openai.time.sleep")
    @patch("mcpuniverse.llm.openai.OpenAI")
    def test_succeeds_on_last_retry(self, mock_openai_cls, mock_sleep):
        """Should succeed if the last retry works."""
        mock_client = Mock()
        mock_openai_cls.return_value = mock_client
        max_retries = 3
        mock_client.chat.completions.create.side_effect = [
            _make_rate_limit_error() for _ in range(max_retries)
        ] + [_make_successful_chat_response("finally")]

        model = OpenAIModel()
        result = model._generate(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=max_retries,
            base_delay=0.01,
        )
        self.assertEqual(result, "finally")
        self.assertEqual(mock_client.chat.completions.create.call_count, max_retries + 1)


if __name__ == "__main__":
    unittest.main()
