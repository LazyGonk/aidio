"""Tests for the web UI module."""

import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aidio.web import _jobs, app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestWebUI:
    """Test the web interface."""

    def test_index_page(self, client):
        """Test that the index page loads."""
        response = client.get("/")
        assert response.status_code == 200
        assert "Aidio" in response.text
        # Assets are versioned so browsers do not keep stale CSS/JS after upgrades.
        assert "style.css?v=" in response.text
        assert "app.js?v=" in response.text

    def test_analyze_success(self, client):
        """Test successful audio analysis."""
        # Create a mock audio file
        audio_data = BytesIO(b"fake audio data")
        audio_data.name = "test.wav"

        mock_result = MagicMock()
        mock_result.chunk_count = 1
        mock_result.total_duration_seconds = 5.0
        mock_result.full_text = "Analysis result"

        with patch("aidio.web.analyze_audio") as mock_analyze:
            mock_analyze.return_value = mock_result

            response = client.post(
                "/analyze",
                files={"audio_file": ("test.wav", audio_data, "audio/wav")},
                data={"prompt": "Test prompt"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["chunk_count"] == 1
            assert data["total_duration"] == 5.0
            assert "Analysis result" in data["analysis"]

    def test_analyze_with_error(self, client):
        """Test error handling during analysis."""
        audio_data = BytesIO(b"fake audio data")
        audio_data.name = "test.wav"

        with patch("aidio.web.analyze_audio") as mock_analyze:
            mock_analyze.side_effect = Exception("Test error")

            response = client.post(
                "/analyze",
                files={"audio_file": ("test.wav", audio_data, "audio/wav")},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "Test error" in data["error"]

    def test_analyze_without_prompt(self, client):
        """Test analysis without custom prompt."""
        audio_data = BytesIO(b"fake audio data")
        audio_data.name = "test.wav"

        mock_result = MagicMock()
        mock_result.chunk_count = 1
        mock_result.total_duration_seconds = 1.0
        mock_result.full_text = "Result"

        with patch("aidio.web.analyze_audio") as mock_analyze:
            mock_analyze.return_value = mock_result

            response = client.post(
                "/analyze",
                files={"audio_file": ("test.wav", audio_data, "audio/wav")},
                data={"prompt": ""},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_analyze_keeps_uploaded_file_suffix(self, client):
        """MP3 uploads must not be staged as .wav temp files."""
        audio_data = BytesIO(b"fake audio data")
        audio_data.name = "song.mp3"

        mock_result = MagicMock()
        mock_result.chunk_count = 1
        mock_result.total_duration_seconds = 1.0
        mock_result.full_text = "Result"

        staged_suffixes = []

        def fake_analyze(path, **kwargs):
            staged_suffixes.append(Path(path).suffix)
            return mock_result

        with patch("aidio.web.analyze_audio", side_effect=fake_analyze):
            response = client.post(
                "/analyze",
                files={"audio_file": ("song.mp3", audio_data, "audio/mpeg")},
            )

        assert response.status_code == 200
        assert staged_suffixes == [".mp3"]


class TestTranscribeEndpoints:
    """Tests for the /transcribe job flow and the embed endpoint."""

    def _transcribe_post(self, client, **extra):
        audio_data = BytesIO(b"fake audio bytes")
        audio_data.name = "song.mp3"
        files = {"audio_file": ("song.mp3", audio_data, "audio/mpeg")}
        data = {
            "device": "cpu",
            "model": "large-v3-turbo",
            "separate_vocals": "false",
            "timestamps": "false",
            "llm_correct": "false",
            "correction_context": "",
        }
        data.update(extra)
        return client.post("/transcribe", files=files, data=data)

    def test_transcribe_rejects_unknown_device(self, client):
        audio_data = BytesIO(b"fake audio bytes")
        audio_data.name = "song.mp3"
        response = client.post(
            "/transcribe",
            files={"audio_file": ("s.mp3", audio_data, "audio/mpeg")},
            data={"device": "vulkan"},
        )
        assert response.status_code == 400
        assert "vulkan" in response.json()["detail"]

    def test_transcribe_rejects_rocm_on_windows(self, client, monkeypatch):
        # rocm is valid on Linux; simulate Windows so validation rejects it.
        monkeypatch.setattr("aidio.lyrics.device.sys.platform", "win32")
        audio_data = BytesIO(b"fake audio bytes")
        audio_data.name = "song.mp3"
        response = client.post(
            "/transcribe",
            files={"audio_file": ("s.mp3", audio_data, "audio/mpeg")},
            data={"device": "rocm"},
        )
        assert response.status_code == 400
        assert "Linux" in response.json()["detail"]

    def test_job_flow(self, client):
        with patch("aidio.pipeline.transcribe_lyrics") as mock_tl:
            from aidio.lyrics.asr import Segment

            mock_tl.return_value = MagicMock(
                lyrics="verse line",
                segments=[Segment(start=0.0, end=1.0, text="verse line")],
                device="cpu",
                used_demucs=False,
                corrections=None,
            )
            response = self._transcribe_post(client)
            assert response.status_code == 200
            job_id = response.json()["job_id"]
            job = client.get(f"/jobs/{job_id}").json()
            assert job["status"] == "done"
            assert job["result"]["lyrics"] == "verse line"
            assert job["result"]["segments"][0]["text"] == "verse line"

    def test_transcribe_forwards_correction_notes(self, client):
        with patch("aidio.pipeline.transcribe_lyrics") as mock_tl:
            mock_tl.return_value = MagicMock(lyrics="x")
            response = self._transcribe_post(
                client, llm_correct="true", correction_context="spell Tool correctly"
            )
            assert response.status_code == 200
            assert mock_tl.call_args.kwargs["correction_context"] == "spell Tool correctly"

    def test_job_error_reported(self, client):
        with patch("aidio.pipeline.transcribe_lyrics", side_effect=RuntimeError("boom")):
            response = self._transcribe_post(client)
            job_id = response.json()["job_id"]
            job = client.get(f"/jobs/{job_id}").json()
            assert job["status"] == "error"
            assert "boom" in job["error"]

    def test_unknown_job_404(self, client):
        response = client.get("/jobs/deadbeef")
        assert response.status_code == 404

    def test_embed_returns_file(self, client):
        audio_data = BytesIO(b"fake mp3 bytes")
        audio_data.name = "song.mp3"
        with patch("aidio.pipeline.transcribe_lyrics") as mock_tl:
            mock_tl.return_value = MagicMock(lyrics="x")
            response = client.post(
                "/transcribe/embed",
                files={"audio_file": ("song.mp3", audio_data, "audio/mpeg")},
                data={"device": "cpu"},
            )
        assert response.status_code == 200
        assert response.content == b"fake mp3 bytes"
        assert "song-lyrics.mp3" in response.headers.get("content-disposition", "")

    def test_embed_endpoint_uses_edited_lyrics(self, client):
        audio_data = BytesIO(b"fake mp3 bytes")
        audio_data.name = "song.mp3"
        with patch("aidio.lyrics.output.embed_lyrics") as mock_embed:
            response = client.post(
                "/embed",
                files={"audio_file": ("song.mp3", audio_data, "audio/mpeg")},
                data={"lyrics": "edited line 1\nedited line 2"},
            )
        assert response.status_code == 200
        assert response.content == b"fake mp3 bytes"
        assert mock_embed.call_args.args[0] == "edited line 1\nedited line 2"
        assert "song-lyrics.mp3" in response.headers.get("content-disposition", "")

    def test_embed_endpoint_reports_unsupported_format(self, client):
        audio_data = BytesIO(b"fake wav bytes")
        audio_data.name = "song.wav"
        response = client.post(
            "/embed",
            files={"audio_file": ("song.wav", audio_data, "audio/wav")},
            data={"lyrics": "line"},
        )
        assert response.status_code == 400
        assert "not supported" in response.json()["detail"]

    def test_embed_rejects_bad_device(self, client):
        audio_data = BytesIO(b"fake mp3 bytes")
        audio_data.name = "song.mp3"
        response = client.post(
            "/transcribe/embed",
            files={"audio_file": ("s.mp3", audio_data, "audio/mpeg")},
            data={"device": "vulkan"},
        )
        assert response.status_code == 400

    def test_device_endpoint(self, client):
        response = client.get("/device")
        assert response.status_code == 200
        assert response.json()["device"] in {"cuda", "cpu", "rocm"}


class TestSettingsAndExplain:
    """Cookie-based settings, explanation endpoint, notices."""

    def test_get_settings_defaults(self, client):
        response = client.get("/settings")
        assert response.status_code == 200
        data = response.json()
        assert "endpoint" in data
        assert "llm_model" in data
        assert "persist" in data

    def test_post_settings_sets_cookie(self, client):
        response = client.post(
            "/settings",
            json={"endpoint": "http://llm:1234/v1", "api_key": "sk-x", "llm_model": "test-model"},
        )
        assert response.status_code == 200
        assert "aidio_settings" in response.headers.get("set-cookie", "")
        assert response.json()["endpoint"] == "http://llm:1234/v1"

    def test_cookie_overrides_env(self, client, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://env-host:9999/v1")
        client.post(
            "/settings",
            json={
                "endpoint": "http://cookie-host:1111/v1",
                "llm_model": "cookie-model",
                "api_key": "cookie-key",
            },
        )
        effective = client.get("/settings/effective").json()
        assert effective["endpoint"] == "http://cookie-host:1111/v1"
        assert effective["llm_model"] == "cookie-model"
        assert effective["api_key"] == "•••"

    def test_no_cookie_falls_back_to_env(self, client, monkeypatch):
        fresh = TestClient(app)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://env-host:9999/v1")
        data = fresh.get("/settings/effective").json()
        assert data["endpoint"] == "http://env-host:9999/v1"

    def test_malformed_cookie_falls_back(self, client):
        fresh = TestClient(app)
        fresh.cookies.set("aidio_settings", "%%%not-base64%%%")
        data = fresh.get("/settings/effective").json()
        assert "endpoint" in data

    def test_analyze_uses_cookie_settings(self, client):
        client.post(
            "/settings",
            json={"endpoint": "http://cookie-llm:1234/v1"},
        )
        audio_data = BytesIO(b"fake audio data")
        audio_data.name = "test.wav"
        with (
            patch("aidio.client.AnalysisClient") as mock_client,
            patch("aidio.web.analyze_audio") as mock_analyze,
        ):
            mock_client.return_value = MagicMock()
            mock_analyze.return_value = MagicMock(
                chunk_count=1, total_duration_seconds=1.0, full_text="ok"
            )
            response = client.post("/analyze", files={"audio_file": ("t.wav", audio_data)})
            assert response.json()["success"] is True
            assert mock_client.call_args.kwargs["base_url"] == "http://cookie-llm:1234/v1"

    def test_explain_success(self, client):
        with patch("aidio.lyrics.explain.explain_lyrics") as mock_ex:
            mock_ex.return_value = MagicMock(
                title="Song",
                themes=["war"],
                motifs=["blade"],
                characters=["hero"],
                narrative_arc="arc",
                worldbuilding_notes=["n"],
                suspicious_lines=[],
                tone="grim",
            )
            response = client.post("/explain", json={"lyrics": "some lyrics"})
            assert response.status_code == 200
            data = response.json()
            assert data["title"] == "Song"
            assert data["themes"] == ["war"]
            assert data["tone"] == "grim"

    def test_correct_endpoint_uses_notes(self, client):
        with patch("aidio.lyrics.correct.correct_lyrics") as mock_correct:
            mock_correct.return_value = "Fixed lyrics"
            response = client.post(
                "/correct", json={"lyrics": "raw lyrics", "notes": "spell Tool correctly"}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["lyrics"] == "Fixed lyrics"
            assert data["changed"] is True
            assert mock_correct.call_args.kwargs["context"] == "spell Tool correctly"

    def test_correct_endpoint_reports_unchanged(self, client):
        with patch("aidio.lyrics.correct.correct_lyrics", return_value="raw lyrics"):
            response = client.post("/correct", json={"lyrics": "raw lyrics"})
        assert response.json()["changed"] is False

    def test_notices_endpoint(self, client):
        response = client.get("/notices")
        assert response.status_code == 200
        assert "Third-Party Notices" in response.text
        # at least one package row present (fastapi, etc.)
        assert "fastapi" in response.text

    def test_lyrics_txt_download(self, client):
        audio_data = BytesIO(b"fake audio bytes")
        audio_data.name = "my_song.mp3"
        with patch("aidio.pipeline.transcribe_lyrics") as mock_tl:
            mock_tl.return_value = MagicMock(
                lyrics="line1\nline2",
                segments=[],
                device="cuda",
                used_demucs=False,
                corrections=None,
            )
            response = client.post(
                "/transcribe",
                files={"audio_file": ("my_song.mp3", audio_data, "audio/mpeg")},
                data={"device": "cuda"},
            )
            job_id = response.json()["job_id"]
            txt = client.get(f"/jobs/{job_id}/lyrics.txt")
            assert txt.status_code == 200
            assert 'filename="my_song.txt"' in txt.headers["content-disposition"]
            assert txt.text == "line1\nline2"

    def test_lyrics_txt_not_finished_409(self, client):
        audio_data = BytesIO(b"x")
        audio_data.name = "s.mp3"
        with patch("aidio.pipeline.transcribe_lyrics", side_effect=RuntimeError("x")):
            response = client.post(
                "/transcribe",
                files={"audio_file": ("s.mp3", audio_data, "audio/mpeg")},
                data={"device": "cpu"},
            )
            job_id = response.json()["job_id"]
            assert client.get(f"/jobs/{job_id}/lyrics.txt").status_code == 409


class TestCancelAndBundle:
    """Job cancellation and the multi-result zip download."""

    def test_cancel_unknown_job_404(self, client):
        assert client.post("/jobs/deadbeef/cancel").status_code == 404

    def test_cancel_running_job_sets_flag(self, client):
        _jobs["cancel-me"] = {"status": "running", "progress": 10.0, "stage": "Transcribing"}
        try:
            response = client.post("/jobs/cancel-me/cancel")
            assert response.status_code == 200
            assert _jobs["cancel-me"]["cancel_requested"] is True
            assert response.json()["status"] == "cancelling"
        finally:
            _jobs.pop("cancel-me", None)

    def test_cancel_finished_job_conflict(self, client):
        _jobs["finished"] = {"status": "done", "progress": 100.0, "stage": "Done"}
        try:
            assert client.post("/jobs/finished/cancel").status_code == 409
        finally:
            _jobs.pop("finished", None)

    def test_bundle_zip_returns_all_files(self, client):
        response = client.post(
            "/bundle/zip",
            json={
                "files": [
                    {"name": "song one.txt", "text": "line 1"},
                    {"name": "dir/song one.txt", "text": "line 2"},
                    {"name": "", "text": "fallback"},
                ]
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        assert "aidio-lyrics.zip" in response.headers["content-disposition"]
        with zipfile.ZipFile(BytesIO(response.content)) as archive:
            assert archive.namelist() == ["song one.txt", "song one-2.txt", "result-3.txt"]
            assert archive.read("song one.txt").decode() == "line 1"

    def test_bundle_zip_rejects_empty(self, client):
        assert client.post("/bundle/zip", json={"files": []}).status_code == 400
