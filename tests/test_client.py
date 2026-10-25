"""Tests for the API client module."""

from unittest.mock import MagicMock, patch

from aidio.client import AnalysisClient


class TestAnalysisClient:
    """Test the OpenAI-compatible client."""

    def test_initializes_with_defaults(self):
        with patch("aidio.client.get_settings") as mock_settings:
            mock_settings.return_value.base_url = "http://test:4000/v1"
            mock_settings.return_value.api_key = "test-key"
            mock_settings.return_value.model = "test-model"

            client = AnalysisClient()

            assert client.base_url == "http://test:4000/v1"
            assert client.api_key == "test-key"
            assert client.model == "test-model"

    def test_initializes_with_overrides(self):
        client = AnalysisClient(
            base_url="http://custom:4000/v1",
            api_key="custom-key",
        )

        assert client.base_url == "http://custom:4000/v1"
        assert client.api_key == "custom-key"

    @patch("aidio.client.get_settings")
    @patch("aidio.client.httpx.Client")
    def test_analyze_sends_correct_payload(self, mock_client_cls, mock_settings):
        mock_settings.return_value.model = "test-model"
        mock_settings.return_value.base_url = "http://test:4000/v1"
        mock_settings.return_value.api_key = "test-key"
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test analysis result"}}]
        }
        mock_client.post.return_value = mock_response

        client = AnalysisClient()
        result = client.analyze("Analyze this", "dGVzdA==")

        assert result == "Test analysis result"

        # Verify the POST call
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/chat/completions"

        json_payload = call_args[1]["json"]
        assert json_payload["model"] == "test-model"
        assert len(json_payload["messages"]) == 1
        assert json_payload["messages"][0]["role"] == "user"
        assert len(json_payload["messages"][0]["content"]) == 2

    def test_close_closes_client(self):
        with patch("aidio.client.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            client = AnalysisClient()
            client.close()

            mock_client.close.assert_called_once()
