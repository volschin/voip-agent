# VoIP Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully local German-language voice AI agent that handles inbound/outbound calls via Fritzbox SIP, using Qwen3-ASR/TTS on DGX Spark and Nous Hermes via vLLM for conversation with RAG (pgvector) and MS Graph calendar tools.

**Architecture:** Asterisk on a NUC registers with Fritzbox via PJSIP and exposes ARI (Asterisk REST Interface). A Python asyncio agent connects via ARI WebSocket + UDP ExternalMedia RTP, runs each voice turn through STT→LLM→TTS, and invokes pgvector RAG and MS Graph calendar as LLM tools. All AI inference runs on a DGX Spark over LAN.

**Tech Stack:** Python 3.12, asyncio, websockets, httpx, asyncpg, pydantic-settings, webrtcvad, scipy/numpy, msal; Asterisk 20 + PJSIP, ARI; Docker Compose (DGX services)

---

## File Map

| File | Responsibility |
|---|---|
| `agent/config.py` | Typed config from env vars via pydantic-settings |
| `agent/session.py` | Per-call state machine (ANSWER→LISTENING→PROCESSING→SPEAKING→ENDED) |
| `agent/audio.py` | G.711 aLaw codec, 8↔16kHz resampling, VAD speech buffer |
| `agent/stt.py` | Async HTTP client → Qwen3-ASR-1.7B |
| `agent/tts.py` | Async HTTP client → Qwen3-TTS |
| `agent/llm.py` | OpenAI-compat chat + tool-call dispatch loop |
| `agent/tools/rag.py` | pgvector cosine search via asyncpg |
| `agent/tools/calendar.py` | CalendarBackend protocol + MSGraphCalendar implementation |
| `agent/pipeline.py` | Orchestrates one voice turn: VAD flush → STT → LLM → TTS |
| `agent/rtp.py` | asyncio UDP DatagramProtocol, RTP packet parse/build |
| `agent/ari.py` | ARI WebSocket events + REST helpers + ExternalMedia setup |
| `agent/main.py` | Entry point: init all clients, start ARI event loop |
| `asterisk/pjsip.conf` | Fritzbox SIP registration |
| `asterisk/extensions.conf` | Route inbound calls to voip-agent Stasis app |
| `asterisk/ari.conf` | ARI HTTP + WebSocket credentials |
| `dgx/docker-compose.yml` | Qwen3-ASR, Qwen3-TTS, embedding sidecar containers |

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `agent/__init__.py`, `agent/tools/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p agent/tools tests asterisk dgx
touch agent/__init__.py agent/tools/__init__.py tests/__init__.py
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=71"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "voip-agent"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "websockets>=12.0",
    "httpx>=0.27",
    "asyncpg>=0.29",
    "msal>=1.28",
    "scipy>=1.13",
    "numpy>=1.26",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "webrtcvad>=2.0.10",
    "aiofiles>=23.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "pytest-asyncio>=0.23",
    "respx>=0.21",
    "pytest-mock>=3.14",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.setuptools.packages.find]
where = ["."]
include = ["agent*"]
```

- [ ] **Step 3: Create `.env.example`**

```bash
# Asterisk ARI
ARI_BASE_URL=http://localhost:8088
ARI_USERNAME=voip-agent
ARI_PASSWORD=changeme
ARI_APP_NAME=voip-agent

# RTP — agent binds here to receive ExternalMedia audio
RTP_BIND_HOST=0.0.0.0
RTP_PORT=5000

# DGX Spark AI services
STT_BASE_URL=http://dgx-spark:8001
TTS_BASE_URL=http://dgx-spark:8002
LLM_BASE_URL=http://dgx-spark:8000
LLM_MODEL=nous-hermes
EMBEDDING_BASE_URL=http://dgx-spark:8003

# pgvector RAG
DB_DSN=postgresql://user:pass@dgx-spark:5432/voip

# MS Graph Calendar
AZURE_TENANT_ID=
AZURE_CLIENT_ID=
AZURE_CLIENT_SECRET=
CALENDAR_USER_EMAIL=

# Agent behaviour
CALLER_ID=+49123456789
GREETING_TEXT=Hallo, wie kann ich Ihnen helfen?
LLM_SYSTEM_PROMPT=Du bist ein hilfreicher Telefonassistent. Antworte immer auf Deutsch. Sei freundlich und präzise. Nutze rag_lookup für Wissensfragen und die Kalender-Werkzeuge für Terminanfragen.
```

- [ ] **Step 4: Create `tests/conftest.py`**

```python
import pytest
from agent.config import Settings


@pytest.fixture
def settings():
    return Settings(
        ari_base_url="http://localhost:8088",
        ari_username="test",
        ari_password="test",
        ari_app_name="voip-agent",
        rtp_bind_host="127.0.0.1",
        rtp_port=5000,
        stt_base_url="http://stt:8001",
        tts_base_url="http://tts:8002",
        llm_base_url="http://llm:8000",
        llm_model="nous-hermes",
        embedding_base_url="http://embed:8003",
        db_dsn="postgresql://u:p@host:5432/db",
        azure_tenant_id="tenant",
        azure_client_id="client",
        azure_client_secret="secret",
        calendar_user_email="user@example.com",
        greeting_text="Hallo!",
        llm_system_prompt="Du bist ein Assistent.",
    )
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -e ".[dev]"
```

Expected: no errors. `pytest --collect-only` shows 0 items.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .env.example agent/ tests/ asterisk/ dgx/
git commit -m "chore: project scaffold"
```

---

## Task 2: Config

**Files:**
- Create: `agent/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py
from agent.config import Settings


def test_settings_load_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("ARI_BASE_URL", raising=False)
    s = Settings(
        ari_base_url="http://test:8088",
        ari_username="u",
        ari_password="p",
        ari_app_name="app",
        rtp_bind_host="0.0.0.0",
        rtp_port=5001,
        stt_base_url="http://stt",
        tts_base_url="http://tts",
        llm_base_url="http://llm",
        llm_model="hermes",
        embedding_base_url="http://emb",
        db_dsn="postgresql://x",
        azure_tenant_id="t",
        azure_client_id="c",
        azure_client_secret="s",
        calendar_user_email="x@x.com",
        greeting_text="Hi!",
        llm_system_prompt="prompt",
    )
    assert s.ari_base_url == "http://test:8088"
    assert s.rtp_port == 5001
    assert s.llm_model == "hermes"
    assert s.greeting_text == "Hi!"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.config'`

- [ ] **Step 3: Implement `agent/config.py`**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ari_base_url: str = "http://localhost:8088"
    ari_username: str = "voip-agent"
    ari_password: str = "changeme"
    ari_app_name: str = "voip-agent"

    rtp_bind_host: str = "0.0.0.0"
    rtp_port: int = 5000

    stt_base_url: str = "http://localhost:8001"
    tts_base_url: str = "http://localhost:8002"
    llm_base_url: str = "http://localhost:8000"
    llm_model: str = "nous-hermes"
    embedding_base_url: str = "http://localhost:8003"

    db_dsn: str = "postgresql://user:pass@localhost:5432/voip"

    azure_tenant_id: str = ""
    azure_client_id: str = ""
    azure_client_secret: str = ""
    calendar_user_email: str = ""

    caller_id: str = ""  # phone number shown on outbound calls, e.g. +49123456789

    greeting_text: str = "Hallo, wie kann ich Ihnen helfen?"
    llm_system_prompt: str = (
        "Du bist ein hilfreicher Telefonassistent. Antworte immer auf Deutsch. "
        "Sei freundlich und präzise. Nutze rag_lookup für Wissensfragen und "
        "die Kalender-Werkzeuge für Terminanfragen."
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add agent/config.py tests/test_config.py
git commit -m "feat: config module with pydantic-settings"
```

---

## Task 3: Call session state machine

**Files:**
- Create: `agent/session.py`
- Create: `tests/test_session.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_session.py
import pytest
from datetime import datetime, timezone
from agent.session import CallSession, SessionState


def _make() -> CallSession:
    return CallSession(
        call_id="ch-123",
        caller_id="+49123456789",
        history=[],
        created_at=datetime.now(timezone.utc),
    )


def test_initial_state_is_answer():
    assert _make().state == SessionState.ANSWER


def test_full_happy_path():
    s = _make()
    s.transition(SessionState.LISTENING)    # greeting finished
    s.transition(SessionState.PROCESSING)  # speech detected
    s.transition(SessionState.SPEAKING)    # TTS ready
    s.transition(SessionState.LISTENING)   # turn complete
    assert s.state == SessionState.LISTENING


def test_interruption():
    s = _make()
    s.transition(SessionState.LISTENING)
    s.transition(SessionState.PROCESSING)
    s.transition(SessionState.SPEAKING)
    s.transition(SessionState.PROCESSING)  # caller speaks mid-playback
    assert s.state == SessionState.PROCESSING


def test_any_state_to_ended():
    for initial in (SessionState.LISTENING, SessionState.PROCESSING, SessionState.SPEAKING):
        s = _make()
        s.state = initial
        s.transition(SessionState.ENDED)
        assert s.state == SessionState.ENDED


def test_invalid_transition_raises():
    s = _make()
    s.transition(SessionState.LISTENING)
    with pytest.raises(ValueError, match="Invalid transition"):
        s.transition(SessionState.SPEAKING)  # LISTENING → SPEAKING not allowed
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_session.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.session'`

- [ ] **Step 3: Implement `agent/session.py`**

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SessionState(str, Enum):
    ANSWER = "answer"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ENDED = "ended"


_VALID = frozenset({
    (SessionState.ANSWER, SessionState.LISTENING),
    (SessionState.LISTENING, SessionState.PROCESSING),
    (SessionState.LISTENING, SessionState.ENDED),
    (SessionState.PROCESSING, SessionState.SPEAKING),
    (SessionState.PROCESSING, SessionState.ENDED),
    (SessionState.SPEAKING, SessionState.LISTENING),
    (SessionState.SPEAKING, SessionState.PROCESSING),
    (SessionState.SPEAKING, SessionState.ENDED),
})


@dataclass
class CallSession:
    call_id: str
    caller_id: str
    history: list[dict]
    created_at: datetime
    state: SessionState = SessionState.ANSWER

    def transition(self, new_state: SessionState) -> None:
        if new_state == SessionState.ENDED:
            self.state = new_state
            return
        if (self.state, new_state) not in _VALID:
            raise ValueError(f"Invalid transition: {self.state} → {new_state}")
        self.state = new_state
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_session.py -v
```

Expected: all 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add agent/session.py tests/test_session.py
git commit -m "feat: call session state machine"
```

---

## Task 4: Audio utilities

**Files:**
- Create: `agent/audio.py`
- Create: `tests/test_audio.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_audio.py
import audioop
import numpy as np
import pytest
from agent.audio import alaw_decode, alaw_encode, resample_8k_to_16k, resample_24k_to_8k, VadBuffer


def _sine_8k(duration_ms: int = 200, freq: int = 440) -> np.ndarray:
    n = int(8000 * duration_ms / 1000)
    t = np.arange(n) / 8000
    return (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)


def test_alaw_roundtrip():
    original = _sine_8k(100)
    encoded = alaw_encode(original)
    decoded = alaw_decode(encoded)
    assert decoded.dtype == np.int16
    assert len(decoded) == len(original)
    # aLaw is lossy — allow ~1% RMS difference
    rms_orig = np.sqrt(np.mean(original.astype(np.float32) ** 2))
    rms_diff = np.sqrt(np.mean((decoded.astype(np.float32) - original.astype(np.float32)) ** 2))
    assert rms_diff / rms_orig < 0.01


def test_resample_8k_to_16k_shape():
    samples = _sine_8k(100)          # 800 samples at 8kHz
    out = resample_8k_to_16k(samples)
    assert len(out) == 1600           # 1600 samples at 16kHz


def test_resample_24k_to_8k_shape():
    n = int(24000 * 0.1)             # 100ms at 24kHz = 2400 samples
    samples = np.zeros(n, dtype=np.int16)
    out = resample_24k_to_8k(samples)
    assert len(out) == 800            # 100ms at 8kHz = 800 samples


def test_vad_buffer_returns_none_during_silence():
    buf = VadBuffer(sample_rate=16000, frame_ms=20, silence_threshold_ms=200)
    frame = np.zeros(320, dtype=np.int16)  # 20ms silence at 16kHz
    for _ in range(20):
        result = buf.add_frame(frame)
    assert result is None


def test_vad_buffer_flushes_after_speech_then_silence():
    buf = VadBuffer(sample_rate=16000, frame_ms=20, silence_threshold_ms=200)
    # Loud sine = speech
    t = np.arange(320) / 16000
    speech_frame = (np.sin(2 * np.pi * 440 * t) * 30000).astype(np.int16)
    silence_frame = np.zeros(320, dtype=np.int16)

    for _ in range(15):  # 300ms speech
        buf.add_frame(speech_frame)

    result = None
    for _ in range(15):  # 300ms silence → exceeds 200ms threshold
        result = buf.add_frame(silence_frame)
        if result is not None:
            break

    assert result is not None
    assert result.dtype == np.int16
    assert len(result) > 0


def test_vad_buffer_hard_cap_flushes_at_15s():
    """VadBuffer must flush at max_speech_ms even without trailing silence."""
    buf = VadBuffer(sample_rate=16000, frame_ms=20, silence_threshold_ms=800, max_speech_ms=200)
    t = np.arange(320) / 16000
    speech_frame = (np.sin(2 * np.pi * 440 * t) * 30000).astype(np.int16)

    results = []
    for _ in range(15):  # 300ms > 200ms cap
        r = buf.add_frame(speech_frame)
        if r is not None:
            results.append(r)

    assert len(results) == 1  # exactly one flush at cap
    assert results[0].dtype == np.int16
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_audio.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.audio'`

- [ ] **Step 3: Implement `agent/audio.py`**

```python
import audioop  # built-in, Python ≤3.12; provides G.711 codec

import numpy as np
import webrtcvad
from scipy.signal import resample_poly


def alaw_decode(data: bytes) -> np.ndarray:
    """G.711 aLaw bytes → int16 PCM array (8 kHz)."""
    return np.frombuffer(audioop.alaw2lin(data, 2), dtype=np.int16)


def alaw_encode(samples: np.ndarray) -> bytes:
    """int16 PCM array (8 kHz) → G.711 aLaw bytes."""
    return audioop.lin2alaw(samples.astype(np.int16).tobytes(), 2)


def resample_8k_to_16k(samples: np.ndarray) -> np.ndarray:
    """Upsample from 8 kHz to 16 kHz (×2)."""
    return resample_poly(samples, up=2, down=1).astype(np.int16)


def resample_24k_to_8k(samples: np.ndarray) -> np.ndarray:
    """Downsample from 24 kHz to 8 kHz (÷3)."""
    return resample_poly(samples, up=1, down=3).astype(np.int16)


class VadBuffer:
    """Accumulates 16 kHz PCM frames and yields a speech chunk after trailing silence."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        silence_threshold_ms: int = 800,
        max_speech_ms: int = 15000,
    ) -> None:
        self._vad = webrtcvad.Vad(2)  # aggressiveness 0-3
        self._sample_rate = sample_rate
        self._silence_threshold = silence_threshold_ms // frame_ms
        self._max_speech_frames = max_speech_ms // frame_ms
        self._speech_frames: list[np.ndarray] = []
        self._silence_count = 0
        self._in_speech = False

    def add_frame(self, frame: np.ndarray) -> np.ndarray | None:
        """Add a 20 ms int16 frame. Returns concatenated speech when utterance ends."""
        is_speech = self._vad.is_speech(
            frame.astype(np.int16).tobytes(), self._sample_rate
        )
        if is_speech:
            self._speech_frames.append(frame)
            self._silence_count = 0
            self._in_speech = True
        elif self._in_speech:
            self._speech_frames.append(frame)
            self._silence_count += 1
            if self._silence_count >= self._silence_threshold:
                return self._flush()
        # Hard cap: flush after 15 s to prevent unbounded buffer growth
        if len(self._speech_frames) >= self._max_speech_frames:
            return self._flush()
        return None

    def force_flush(self) -> np.ndarray | None:
        """Return accumulated speech immediately (e.g. on hangup)."""
        if self._speech_frames:
            return self._flush()
        return None

    def reset(self) -> None:
        self._speech_frames = []
        self._silence_count = 0
        self._in_speech = False

    def _flush(self) -> np.ndarray:
        result = np.concatenate(self._speech_frames)
        self.reset()
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_audio.py -v
```

Expected: all 6 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add agent/audio.py tests/test_audio.py
git commit -m "feat: audio utilities — aLaw codec, resampling, VAD buffer"
```

---

## Task 5: STT client

**Files:**
- Create: `agent/stt.py`
- Create: `tests/test_stt.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_stt.py
import io
import wave
import pytest
import respx
import httpx
import numpy as np
from agent.stt import SttClient


@pytest.fixture
def stt(settings):
    return SttClient(base_url=settings.stt_base_url)


def _pcm_16k(duration_ms: int = 500) -> bytes:
    n = 16000 * duration_ms // 1000
    return (np.zeros(n, dtype=np.int16)).tobytes()


@respx.mock
async def test_transcribe_returns_text(stt):
    respx.post("http://stt:8001/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "Hallo Welt"})
    )
    result = await stt.transcribe(_pcm_16k())
    assert result == "Hallo Welt"


@respx.mock
async def test_transcribe_sends_wav_with_language(stt):
    route = respx.post("http://stt:8001/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "test"})
    )
    await stt.transcribe(_pcm_16k())
    request = route.calls[0].request
    # Multipart body must contain language=de
    assert b"language" in request.content
    assert b"de" in request.content


@respx.mock
async def test_transcribe_raises_on_http_error(stt):
    respx.post("http://stt:8001/v1/audio/transcriptions").mock(
        return_value=httpx.Response(500, text="Server error")
    )
    with pytest.raises(httpx.HTTPStatusError):
        await stt.transcribe(_pcm_16k())
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_stt.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.stt'`

- [ ] **Step 3: Implement `agent/stt.py`**

```python
import io
import wave

import httpx
import numpy as np


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class SttClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def transcribe(self, pcm_16k: bytes) -> str:
        """Send 16 kHz PCM bytes to Qwen3-ASR and return transcript."""
        wav = _pcm_to_wav(pcm_16k)
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base_url}/v1/audio/transcriptions",
                files={"file": ("audio.wav", wav, "audio/wav")},
                data={"language": "de"},
                timeout=30.0,
            )
        resp.raise_for_status()
        return resp.json()["text"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_stt.py -v
```

Expected: all 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add agent/stt.py tests/test_stt.py
git commit -m "feat: STT client for Qwen3-ASR"
```

---

## Task 6: TTS client

**Files:**
- Create: `agent/tts.py`
- Create: `tests/test_tts.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tts.py
import pytest
import respx
import httpx
import numpy as np
from agent.tts import TtsClient


@pytest.fixture
def tts(settings):
    return TtsClient(base_url=settings.tts_base_url)


def _fake_pcm(n_samples: int = 24000) -> bytes:
    return (np.zeros(n_samples, dtype=np.int16)).tobytes()


@respx.mock
async def test_synthesize_returns_pcm(tts):
    fake_audio = _fake_pcm()
    respx.post("http://tts:8002/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=fake_audio)
    )
    result = await tts.synthesize("Hallo Welt")
    assert result == fake_audio


@respx.mock
async def test_synthesize_sends_text_and_voice(tts):
    route = respx.post("http://tts:8002/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=_fake_pcm())
    )
    await tts.synthesize("Test")
    body = route.calls[0].request.read()
    import json
    payload = json.loads(body)
    assert payload["input"] == "Test"
    assert "instruct" in payload


@respx.mock
async def test_synthesize_raises_on_http_error(tts):
    respx.post("http://tts:8002/v1/audio/speech").mock(
        return_value=httpx.Response(503, text="unavailable")
    )
    with pytest.raises(httpx.HTTPStatusError):
        await tts.synthesize("Test")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tts.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.tts'`

- [ ] **Step 3: Implement `agent/tts.py`**

```python
import httpx


VOICE_INSTRUCT = (
    "Eine warme, natürliche deutsche Stimme mit angemessenem Sprechtempo."
)


class TtsClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def synthesize(self, text: str) -> bytes:
        """Send text to Qwen3-TTS and return raw PCM bytes (24 kHz, mono, int16)."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base_url}/v1/audio/speech",
                json={"input": text, "instruct": VOICE_INSTRUCT},
                timeout=30.0,
            )
        resp.raise_for_status()
        return resp.content
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tts.py -v
```

Expected: all 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add agent/tts.py tests/test_tts.py
git commit -m "feat: TTS client for Qwen3-TTS"
```

---

## Task 7: LLM client with tool-call dispatch

**Files:**
- Create: `agent/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm.py
import json
import pytest
import respx
import httpx
from unittest.mock import AsyncMock
from agent.llm import LlmClient


@pytest.fixture
def llm(settings):
    rag = AsyncMock(return_value="RAG result")
    calendar = AsyncMock()
    calendar.get_events = AsyncMock(return_value="No events")
    calendar.create_event = AsyncMock(return_value="Event created")
    return LlmClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        system_prompt=settings.llm_system_prompt,
        rag=rag,
        calendar=calendar,
    )


def _chat_response(content: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content, "tool_calls": None}}]
    }


def _tool_call_response(name: str, arguments: dict) -> dict:
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }],
            }
        }]
    }


@respx.mock
async def test_complete_no_tool(llm):
    respx.post("http://llm:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response("Hallo!"))
    )
    result = await llm.complete([{"role": "user", "content": "Hi"}])
    assert result == "Hallo!"


@respx.mock
async def test_complete_with_rag_tool_call(llm):
    responses = [
        httpx.Response(200, json=_tool_call_response("rag_lookup", {"query": "Was ist X?"})),
        httpx.Response(200, json=_chat_response("X ist Y.")),
    ]
    respx.post("http://llm:8000/v1/chat/completions").mock(side_effect=responses)
    result = await llm.complete([{"role": "user", "content": "Was ist X?"}])
    assert result == "X ist Y."
    llm._rag.assert_awaited_once_with("Was ist X?")


@respx.mock
async def test_complete_raises_on_http_error(llm):
    respx.post("http://llm:8000/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="error")
    )
    with pytest.raises(httpx.HTTPStatusError):
        await llm.complete([{"role": "user", "content": "test"}])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_llm.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.llm'`

- [ ] **Step 3: Implement `agent/llm.py`**

```python
import json
from typing import Any, Protocol

import httpx


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_lookup",
            "description": "Durchsuche die Wissensdatenbank nach relevanten Informationen.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_get_events",
            "description": "Kalendertermine für einen Zeitraum abrufen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "ISO 8601 datetime"},
                    "end": {"type": "string", "description": "ISO 8601 datetime"},
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_create_event",
            "description": "Neuen Kalendertermin erstellen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "string", "description": "ISO 8601 datetime"},
                    "end": {"type": "string", "description": "ISO 8601 datetime"},
                    "description": {"type": "string", "default": ""},
                },
                "required": ["title", "start", "end"],
            },
        },
    },
]


class LlmClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        system_prompt: str,
        rag: Any,
        calendar: Any,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._system_prompt = system_prompt
        self._rag = rag
        self._calendar = calendar

    async def complete(self, messages: list[dict]) -> str:
        """Run chat completion with tool-call loop. Returns final text response."""
        full_messages = [
            {"role": "system", "content": self._system_prompt},
            *messages,
        ]
        async with httpx.AsyncClient() as client:
            while True:
                resp = await client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json={"model": self._model, "messages": full_messages, "tools": TOOLS},
                    timeout=60.0,
                )
                resp.raise_for_status()
                msg = resp.json()["choices"][0]["message"]

                if not msg.get("tool_calls"):
                    return msg["content"] or ""

                full_messages.append(msg)
                for tc in msg["tool_calls"]:
                    result = await self._dispatch(
                        tc["function"]["name"],
                        json.loads(tc["function"]["arguments"]),
                    )
                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

    async def _dispatch(self, name: str, args: dict) -> str:
        if name == "rag_lookup":
            return await self._rag(args["query"])
        if name == "calendar_get_events":
            return await self._calendar.get_events(args["start"], args["end"])
        if name == "calendar_create_event":
            return await self._calendar.create_event(
                title=args["title"],
                start=args["start"],
                end=args["end"],
                description=args.get("description", ""),
            )
        return f"Unknown tool: {name}"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_llm.py -v
```

Expected: all 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add agent/llm.py tests/test_llm.py
git commit -m "feat: LLM client with OpenAI-compatible tool-call loop"
```

---

## Task 8: RAG tool

**Files:**
- Create: `agent/tools/rag.py`
- Create: `tests/test_tools_rag.py`

Database schema expected (run once on pgvector DB):
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1024)
);
```

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tools_rag.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agent.tools.rag import RagTool


@pytest.fixture
def pool():
    mock_pool = AsyncMock()
    conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_pool, conn


async def test_lookup_returns_joined_chunks(settings, pool):
    mock_pool, conn = pool
    # fetch returns embedding vector
    embedding_response = MagicMock()
    embedding_response.json = MagicMock(return_value={"embedding": [0.1] * 1024})

    conn.fetch.return_value = [
        {"content": "Chunk A"},
        {"content": "Chunk B"},
    ]

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_http = AsyncMock()
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        mock_http.post = AsyncMock(return_value=embedding_response)
        mock_client_cls.return_value = mock_http

        rag = RagTool(pool=mock_pool, embedding_base_url=settings.embedding_base_url)
        result = await rag.lookup("Was ist X?")

    assert "Chunk A" in result
    assert "Chunk B" in result


async def test_lookup_empty_result(settings, pool):
    mock_pool, conn = pool
    embedding_response = MagicMock()
    embedding_response.json = MagicMock(return_value={"embedding": [0.0] * 1024})
    conn.fetch.return_value = []

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_http = AsyncMock()
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        mock_http.post = AsyncMock(return_value=embedding_response)
        mock_client_cls.return_value = mock_http

        rag = RagTool(pool=mock_pool, embedding_base_url=settings.embedding_base_url)
        result = await rag.lookup("unbekannt")

    assert result == "Keine relevanten Informationen gefunden."
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tools_rag.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.tools.rag'`

- [ ] **Step 3: Implement `agent/tools/rag.py`**

```python
import httpx
import asyncpg

TOP_K = 5


class RagTool:
    def __init__(self, pool: asyncpg.Pool, embedding_base_url: str) -> None:
        self._pool = pool
        self._embed_url = embedding_base_url.rstrip("/") + "/embed"

    async def lookup(self, query: str) -> str:
        """Embed query, search pgvector, return top-k chunks as text."""
        embedding = await self._embed(query)
        rows = await self._search(embedding)
        if not rows:
            return "Keine relevanten Informationen gefunden."
        return "\n\n".join(row["content"] for row in rows)

    async def _embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self._embed_url,
                json={"text": text},
                timeout=10.0,
            )
        resp.raise_for_status()
        return resp.json()["embedding"]

    async def _search(self, embedding: list[float]) -> list[asyncpg.Record]:
        vec_literal = "[" + ",".join(str(x) for x in embedding) + "]"
        async with self._pool.acquire() as conn:
            return await conn.fetch(
                """
                SELECT content
                FROM documents
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                vec_literal,
                TOP_K,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tools_rag.py -v
```

Expected: all 2 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add agent/tools/rag.py tests/test_tools_rag.py
git commit -m "feat: pgvector RAG tool"
```

---

## Task 9: Calendar tool (MS Graph)

**Files:**
- Create: `agent/tools/calendar.py`
- Create: `tests/test_tools_calendar.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tools_calendar.py
import pytest
import respx
import httpx
from unittest.mock import MagicMock
from agent.tools.calendar import MSGraphCalendar


@pytest.fixture
def calendar(settings):
    msal_app = MagicMock()
    msal_app.acquire_token_for_client.return_value = {"access_token": "tok123"}
    return MSGraphCalendar(
        msal_app=msal_app,
        user_email=settings.calendar_user_email,
    )


@respx.mock
async def test_get_events_returns_formatted_string(calendar):
    respx.get(
        f"https://graph.microsoft.com/v1.0/users/{calendar._user_email}/calendarView"
    ).mock(return_value=httpx.Response(200, json={
        "value": [{
            "subject": "Team Meeting",
            "start": {"dateTime": "2026-05-25T10:00:00"},
            "end": {"dateTime": "2026-05-25T11:00:00"},
        }]
    }))
    result = await calendar.get_events("2026-05-25T00:00:00", "2026-05-25T23:59:59")
    assert "Team Meeting" in result
    assert "10:00" in result


@respx.mock
async def test_get_events_no_events(calendar):
    respx.get(
        f"https://graph.microsoft.com/v1.0/users/{calendar._user_email}/calendarView"
    ).mock(return_value=httpx.Response(200, json={"value": []}))
    result = await calendar.get_events("2026-05-25T00:00:00", "2026-05-25T23:59:59")
    assert result == "Keine Termine in diesem Zeitraum."


@respx.mock
async def test_create_event_returns_confirmation(calendar):
    respx.post(
        f"https://graph.microsoft.com/v1.0/users/{calendar._user_email}/events"
    ).mock(return_value=httpx.Response(201, json={"id": "evt1", "subject": "Arzt"}))
    result = await calendar.create_event(
        title="Arzt",
        start="2026-05-26T09:00:00",
        end="2026-05-26T09:30:00",
    )
    assert "Arzt" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tools_calendar.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.tools.calendar'`

- [ ] **Step 3: Implement `agent/tools/calendar.py`**

```python
from typing import Protocol
import httpx


GRAPH_BASE = "https://graph.microsoft.com/v1.0"


class CalendarBackend(Protocol):
    async def get_events(self, start: str, end: str) -> str: ...
    async def create_event(self, title: str, start: str, end: str, description: str = "") -> str: ...


class MSGraphCalendar:
    def __init__(self, msal_app, user_email: str) -> None:
        self._msal = msal_app
        self._user_email = user_email
        self._scope = ["https://graph.microsoft.com/.default"]

    def _token(self) -> str:
        result = self._msal.acquire_token_for_client(scopes=self._scope)
        return result["access_token"]

    async def get_events(self, start: str, end: str) -> str:
        headers = {"Authorization": f"Bearer {self._token()}"}
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{GRAPH_BASE}/users/{self._user_email}/calendarView",
                params={"startDateTime": start, "endDateTime": end,
                        "$select": "subject,start,end"},
                headers=headers,
                timeout=15.0,
            )
        resp.raise_for_status()
        events = resp.json().get("value", [])
        if not events:
            return "Keine Termine in diesem Zeitraum."
        lines = []
        for e in events:
            dt = e["start"]["dateTime"][:16].replace("T", " ")
            lines.append(f"- {e['subject']} um {dt.split()[1]} Uhr")
        return "\n".join(lines)

    async def create_event(self, title: str, start: str, end: str, description: str = "") -> str:
        headers = {
            "Authorization": f"Bearer {self._token()}",
            "Content-Type": "application/json",
        }
        body = {
            "subject": title,
            "body": {"contentType": "text", "content": description},
            "start": {"dateTime": start, "timeZone": "Europe/Berlin"},
            "end": {"dateTime": end, "timeZone": "Europe/Berlin"},
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{GRAPH_BASE}/users/{self._user_email}/events",
                json=body,
                headers=headers,
                timeout=15.0,
            )
        resp.raise_for_status()
        subject = resp.json().get("subject", title)
        return f"Termin '{subject}' wurde erstellt."
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tools_calendar.py -v
```

Expected: all 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add agent/tools/calendar.py tests/test_tools_calendar.py
git commit -m "feat: MS Graph calendar tool with CalendarBackend protocol"
```

---

## Task 10: Pipeline orchestrator

**Files:**
- Create: `agent/pipeline.py`
- Create: `tests/test_pipeline.py`

The pipeline owns one voice turn: receives a 16 kHz PCM chunk from VAD, calls STT → LLM → TTS, returns 8 kHz aLaw bytes for Asterisk.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline.py
import numpy as np
import pytest
from unittest.mock import AsyncMock
from agent.pipeline import VoicePipeline
from agent.session import CallSession, SessionState
from datetime import datetime, timezone


def _make_session() -> CallSession:
    s = CallSession(
        call_id="ch-1",
        caller_id="+49",
        history=[],
        created_at=datetime.now(timezone.utc),
    )
    s.state = SessionState.LISTENING
    return s


def _pcm_16k(duration_ms: int = 200) -> np.ndarray:
    return np.zeros(16000 * duration_ms // 1000, dtype=np.int16)


@pytest.fixture
def pipeline():
    stt = AsyncMock(return_value="Wie ist das Wetter?")
    tts = AsyncMock(return_value=np.zeros(8000, dtype=np.int16).tobytes())
    llm = AsyncMock(return_value="Es ist sonnig.")
    return VoicePipeline(stt=stt, llm=llm, tts=tts)


async def test_process_full_turn_returns_alaw(pipeline):
    session = _make_session()
    pcm_chunk = _pcm_16k(500)
    result = await pipeline.process_turn(session, pcm_chunk)
    assert isinstance(result, bytes)
    assert len(result) > 0
    assert session.state == SessionState.LISTENING


async def test_process_turn_appends_to_history(pipeline):
    session = _make_session()
    await pipeline.process_turn(session, _pcm_16k(200))
    assert len(session.history) == 2  # user + assistant
    assert session.history[0]["role"] == "user"
    assert session.history[1]["role"] == "assistant"


async def test_process_turn_state_machine(pipeline):
    session = _make_session()
    assert session.state == SessionState.LISTENING
    await pipeline.process_turn(session, _pcm_16k(200))
    assert session.state == SessionState.LISTENING  # back to LISTENING after full turn
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.pipeline'`

- [ ] **Step 3: Implement `agent/pipeline.py`**

```python
import logging

import numpy as np

from agent.audio import alaw_encode, resample_24k_to_8k
from agent.session import CallSession, SessionState

log = logging.getLogger(__name__)

# Pre-encoded silent 20ms aLaw frame used as padding
_SILENCE_FRAME = b"\xd5" * 160

FALLBACK_ASR = "Ich habe Sie leider nicht verstanden."
FALLBACK_LLM = "Technischer Fehler, bitte später erneut anrufen."


class VoicePipeline:
    def __init__(self, stt, llm, tts) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts

    async def synthesize_alaw(self, text: str) -> bytes:
        """Synthesize text → 8 kHz aLaw bytes (used for greeting and fallback)."""
        try:
            pcm_24k_bytes = await self._tts.synthesize(text)
            pcm_24k = np.frombuffer(pcm_24k_bytes, dtype=np.int16)
            return alaw_encode(resample_24k_to_8k(pcm_24k))
        except Exception:
            log.exception("TTS failed for text: %r", text[:50])
            return _SILENCE_FRAME  # return minimal audio on TTS failure

    async def process_turn(self, session: CallSession, pcm_16k: np.ndarray) -> bytes:
        """
        Run one voice turn: 16 kHz PCM → transcript → LLM response → aLaw bytes.
        Updates session.history and state. Returns fallback audio on any error.
        """
        session.transition(SessionState.PROCESSING)

        try:
            transcript = await self._stt.transcribe(pcm_16k.tobytes())
        except Exception:
            log.exception("STT failed")
            session.transition(SessionState.LISTENING)
            return await self.synthesize_alaw(FALLBACK_ASR)

        if not transcript.strip():
            session.transition(SessionState.LISTENING)
            return await self.synthesize_alaw(FALLBACK_ASR)

        session.history.append({"role": "user", "content": transcript})

        try:
            response_text = await self._llm.complete(session.history)
        except Exception:
            log.exception("LLM failed")
            session.history.pop()  # remove user message that caused failure
            session.transition(SessionState.LISTENING)
            return await self.synthesize_alaw(FALLBACK_LLM)

        session.history.append({"role": "assistant", "content": response_text})
        session.transition(SessionState.SPEAKING)

        alaw_bytes = await self.synthesize_alaw(response_text)
        session.transition(SessionState.LISTENING)
        return alaw_bytes
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline.py -v
```

Expected: all 3 tests PASSED

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
pytest -v
```

Expected: all tests PASSED

- [ ] **Step 6: Commit**

```bash
git add agent/pipeline.py tests/test_pipeline.py
git commit -m "feat: voice pipeline orchestrator (STT → LLM → TTS)"
```

---

## Task 11: RTP server

**Files:**
- Create: `agent/rtp.py`
- Create: `tests/test_rtp.py`

The RTP server is an asyncio UDP server that strips the 12-byte RTP header from incoming packets and reconstructs RTP packets for outgoing audio.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rtp.py
import asyncio
import struct
import pytest
from unittest.mock import MagicMock
from agent.rtp import RtpServer, parse_rtp_payload, build_rtp_packet


def _rtp_packet(payload: bytes, seq: int = 1, ts: int = 160) -> bytes:
    header = struct.pack("!BBHII", 0x80, 0x08, seq, ts, 0xDEADBEEF)
    return header + payload


def test_parse_rtp_payload_strips_header():
    payload = b"\xd5" * 160  # 160 bytes aLaw
    packet = _rtp_packet(payload)
    result = parse_rtp_payload(packet)
    assert result == payload


def test_parse_rtp_too_short_returns_empty():
    assert parse_rtp_payload(b"\x80\x08") == b""


def test_build_rtp_packet_has_correct_header():
    payload = b"\x00" * 160
    packet = build_rtp_packet(payload, seq=5, timestamp=800, ssrc=0xCAFE)
    assert len(packet) == 172  # 12 header + 160 payload
    v, pt, seq_out, ts_out, ssrc_out = struct.unpack("!BBHII", packet[:12])
    assert v == 0x80
    assert pt == 0x08   # G.711 aLaw
    assert seq_out == 5
    assert ts_out == 800
    assert ssrc_out == 0xCAFE


async def test_rtp_server_calls_callback_with_payload():
    received = []

    def on_audio(payload: bytes) -> None:
        received.append(payload)

    server = RtpServer(host="127.0.0.1", port=0, on_audio=on_audio)
    transport, _ = await asyncio.get_event_loop().create_datagram_endpoint(
        lambda: server, local_addr=("127.0.0.1", 0)
    )
    bound_port = transport.get_extra_info("sockname")[1]

    payload = b"\xd5" * 160
    packet = _rtp_packet(payload)
    send_transport, _ = await asyncio.get_event_loop().create_datagram_endpoint(
        asyncio.DatagramProtocol,
        remote_addr=("127.0.0.1", bound_port),
    )
    send_transport.sendto(packet)
    await asyncio.sleep(0.05)

    transport.close()
    send_transport.close()

    assert received == [payload]


async def test_stream_audio_sends_paced_frames():
    """stream_audio must split audio into 160-byte frames with 20 ms gaps."""
    frames_sent = []

    server = RtpServer(host="127.0.0.1", port=0, on_audio=lambda _: None)
    # Inject a fake transport that records sendto calls
    fake_transport = MagicMock()
    fake_transport.sendto = lambda pkt, _addr: frames_sent.append(pkt)
    server._transport = fake_transport
    server._remote_addr = ("127.0.0.1", 9999)

    # 3 frames of audio
    alaw = b"\xd5" * 480
    await server.stream_audio(alaw)

    assert len(frames_sent) == 3
    # Each RTP packet = 12-byte header + 160-byte payload
    for pkt in frames_sent:
        assert len(pkt) == 172
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_rtp.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.rtp'`

- [ ] **Step 3: Implement `agent/rtp.py`**

```python
import asyncio
import struct
from collections.abc import Callable

RTP_HEADER_SIZE = 12
PAYLOAD_TYPE_ALAW = 0x08


def parse_rtp_payload(packet: bytes) -> bytes:
    if len(packet) < RTP_HEADER_SIZE:
        return b""
    return packet[RTP_HEADER_SIZE:]


def build_rtp_packet(payload: bytes, seq: int, timestamp: int, ssrc: int) -> bytes:
    header = struct.pack(
        "!BBHII",
        0x80,           # V=2, no padding, no ext, no CSRC
        PAYLOAD_TYPE_ALAW,
        seq & 0xFFFF,
        timestamp & 0xFFFFFFFF,
        ssrc & 0xFFFFFFFF,
    )
    return header + payload


class RtpServer(asyncio.DatagramProtocol):
    """Async UDP server that receives RTP from Asterisk ExternalMedia."""

    SAMPLES_PER_FRAME = 160  # 20ms at 8kHz

    def __init__(
        self,
        host: str,
        port: int,
        on_audio: Callable[[bytes], None],
    ) -> None:
        self._host = host
        self._port = port
        self._on_audio = on_audio
        self._transport: asyncio.DatagramTransport | None = None
        self._remote_addr: tuple[str, int] | None = None
        self._ssrc = 0x1234ABCD
        self._seq = 0
        self._timestamp = 0

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        self._transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        if self._remote_addr is None:
            self._remote_addr = addr  # learn Asterisk RTP addr from first packet
        payload = parse_rtp_payload(data)
        if payload:
            self._on_audio(payload)

    def send_frame(self, alaw_frame: bytes) -> None:
        """Send exactly one 160-byte aLaw RTP frame."""
        if not self._transport or not self._remote_addr:
            return
        packet = build_rtp_packet(
            alaw_frame,
            seq=self._seq,
            timestamp=self._timestamp,
            ssrc=self._ssrc,
        )
        self._transport.sendto(packet, self._remote_addr)
        self._seq = (self._seq + 1) & 0xFFFF
        self._timestamp = (self._timestamp + self.SAMPLES_PER_FRAME) & 0xFFFFFFFF

    async def stream_audio(self, alaw: bytes) -> None:
        """Send alaw bytes as paced 20 ms RTP frames. Cancellable via asyncio task."""
        _SILENCE = b"\xd5" * self.SAMPLES_PER_FRAME
        for i in range(0, len(alaw), self.SAMPLES_PER_FRAME):
            chunk = alaw[i : i + self.SAMPLES_PER_FRAME]
            if len(chunk) < self.SAMPLES_PER_FRAME:
                chunk = chunk + _SILENCE[len(chunk):]  # pad last frame
            self.send_frame(chunk)
            await asyncio.sleep(0.02)  # 20 ms per frame

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(
            lambda: self, local_addr=(self._host, self._port)
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_rtp.py -v
```

Expected: all 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add agent/rtp.py tests/test_rtp.py
git commit -m "feat: RTP server for ARI ExternalMedia bidirectional audio"
```

---

## Task 12: ARI client

**Files:**
- Create: `agent/ari.py`
- Create: `tests/test_ari.py`

The ARI client connects to Asterisk's WebSocket event stream, responds to `StasisStart`/`StasisEnd` events, creates an ExternalMedia bridge per call, and wires audio to/from the pipeline.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ari.py
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agent.ari import AriClient
from agent.config import Settings


@pytest.fixture
def ari(settings):
    pipeline = AsyncMock(return_value=b"\xd5" * 160)
    return AriClient(settings=settings, pipeline=pipeline)


def _stasis_start_event(channel_id: str = "ch-1", caller: str = "+49123") -> str:
    return json.dumps({
        "type": "StasisStart",
        "channel": {
            "id": channel_id,
            "caller": {"number": caller},
        },
        "application": "voip-agent",
    })


def _stasis_end_event(channel_id: str = "ch-1") -> str:
    return json.dumps({
        "type": "StasisEnd",
        "channel": {"id": channel_id},
        "application": "voip-agent",
    })


async def test_stasis_start_creates_session(ari):
    with patch.object(ari, "_setup_call", new_callable=AsyncMock) as mock_setup:
        await ari._handle_event(json.loads(_stasis_start_event("ch-1", "+49")))
        mock_setup.assert_awaited_once_with("ch-1", "+49")


async def test_stasis_end_removes_session(ari):
    from agent.session import CallSession, SessionState
    from datetime import datetime, timezone

    session = CallSession(
        call_id="ch-1", caller_id="+49",
        history=[], created_at=datetime.now(timezone.utc)
    )
    session.state = SessionState.LISTENING
    ari._sessions["ch-1"] = session

    await ari._handle_event(json.loads(_stasis_end_event("ch-1")))
    assert "ch-1" not in ari._sessions


async def test_unknown_event_ignored(ari):
    await ari._handle_event({"type": "ChannelDtmfReceived", "channel": {"id": "ch-1"}})
    # no exception
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ari.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.ari'`

- [ ] **Step 3: Implement `agent/ari.py`**

```python
import asyncio
import json
import logging
from datetime import datetime, timezone

import httpx
import websockets

from agent.audio import alaw_decode, resample_8k_to_16k, VadBuffer
from agent.config import Settings
from agent.pipeline import VoicePipeline
from agent.rtp import RtpServer
from agent.session import CallSession, SessionState

log = logging.getLogger(__name__)


class AriClient:
    def __init__(self, settings: Settings, pipeline: VoicePipeline) -> None:
        self._s = settings
        self._pipeline = pipeline
        self._sessions: dict[str, CallSession] = {}
        self._rtp_servers: dict[str, RtpServer] = {}
        self._vad_buffers: dict[str, VadBuffer] = {}
        self._playback_tasks: dict[str, asyncio.Task] = {}  # cancellable per-call playback
        self._rtp_port_counter = settings.rtp_port

    # ── Public entry point ──────────────────────────────────────────────────

    async def run(self) -> None:
        url = (
            f"ws://{self._s.ari_base_url.split('://', 1)[-1]}"
            f"/ari/events?api_key={self._s.ari_username}:{self._s.ari_password}"
            f"&app={self._s.ari_app_name}&subscribeAll=true"
        )
        log.info("Connecting to ARI at %s", url)
        async with websockets.connect(url) as ws:
            async for raw in ws:
                try:
                    event = json.loads(raw)
                    await self._handle_event(event)
                except Exception:
                    log.exception("Error handling ARI event")

    # ── Event handlers ───────────────────────────────────────────────────────

    async def _handle_event(self, event: dict) -> None:
        t = event.get("type")
        if t == "StasisStart":
            ch = event["channel"]
            # Run setup as task so websocket reader is never blocked
            asyncio.create_task(self._setup_call(ch["id"], ch["caller"]["number"]))
        elif t == "StasisEnd":
            ch_id = event["channel"]["id"]
            await self._teardown_call(ch_id)

    # ── Call setup / teardown ────────────────────────────────────────────────

    async def _setup_call(self, channel_id: str, caller_id: str) -> None:
        session = CallSession(
            call_id=channel_id,
            caller_id=caller_id,
            history=[],
            created_at=datetime.now(timezone.utc),
        )
        self._sessions[channel_id] = session

        rtp_port = self._rtp_port_counter
        self._rtp_port_counter += 2

        loop = asyncio.get_running_loop()
        rtp_server = RtpServer(
            host=self._s.rtp_bind_host,
            port=rtp_port,
            on_audio=lambda payload: loop.create_task(
                self._on_audio(channel_id, payload)
            ),
        )
        await rtp_server.start()
        self._rtp_servers[channel_id] = rtp_server
        self._vad_buffers[channel_id] = VadBuffer()

        ext_channel_id = await self._create_external_media(channel_id, rtp_port)
        await self._bridge_channels(channel_id, ext_channel_id)

        # Stream greeting with 20 ms pacing, then enter LISTENING
        alaw = await self._pipeline.synthesize_alaw(self._s.greeting_text)
        await rtp_server.stream_audio(alaw)
        session.transition(SessionState.LISTENING)
        log.info("Call %s from %s ready", channel_id, caller_id)

    async def _teardown_call(self, channel_id: str) -> None:
        task = self._playback_tasks.pop(channel_id, None)
        if task and not task.done():
            task.cancel()
        self._vad_buffers.pop(channel_id, None)
        session = self._sessions.pop(channel_id, None)
        if session:
            session.transition(SessionState.ENDED)
        rtp = self._rtp_servers.pop(channel_id, None)
        if rtp and rtp._transport:
            rtp._transport.close()
        log.info("Call %s ended", channel_id)

    # ── Audio from caller ────────────────────────────────────────────────────

    async def _play_audio(self, channel_id: str, alaw: bytes, session: CallSession) -> None:
        """Stream aLaw audio to caller as paced RTP frames. Cancellable for barge-in."""
        rtp = self._rtp_servers.get(channel_id)
        if not rtp:
            return
        session.transition(SessionState.SPEAKING)
        try:
            await rtp.stream_audio(alaw)
            # Normal completion: back to LISTENING
            if session.state == SessionState.SPEAKING:
                session.transition(SessionState.LISTENING)
            vad = self._vad_buffers.get(channel_id)
            if vad:
                vad.reset()
        except asyncio.CancelledError:
            pass  # barge-in: _on_audio handles state transition

    async def _on_audio(self, channel_id: str, alaw_payload: bytes) -> None:
        session = self._sessions.get(channel_id)
        if not session:
            return

        state = session.state
        if state not in (SessionState.LISTENING, SessionState.SPEAKING):
            return

        vad = self._vad_buffers.get(channel_id)
        if vad is None:
            return

        pcm_8k = alaw_decode(alaw_payload)
        pcm_16k = resample_8k_to_16k(pcm_8k)
        speech = vad.add_frame(pcm_16k)

        if speech is None:
            return

        # Cancel ongoing playback (barge-in or normal turn transition)
        task = self._playback_tasks.pop(channel_id, None)
        if task and not task.done():
            task.cancel()

        # Ensure clean LISTENING state before process_turn
        if session.state == SessionState.SPEAKING:
            session.transition(SessionState.LISTENING)
        vad.reset()

        response_alaw = await self._pipeline.process_turn(session, speech)
        task = asyncio.create_task(self._play_audio(channel_id, response_alaw, session))
        self._playback_tasks[channel_id] = task

    # ── ARI REST helpers ─────────────────────────────────────────────────────

    async def _create_external_media(self, channel_id: str, rtp_port: int) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._s.ari_base_url}/ari/channels/externalMedia",
                auth=(self._s.ari_username, self._s.ari_password),
                params={
                    "app": self._s.ari_app_name,
                    "external_host": f"{self._s.rtp_bind_host}:{rtp_port}",
                    "format": "alaw",
                    "channelId": f"ext-{channel_id}",
                },
                timeout=5.0,
            )
        resp.raise_for_status()
        return resp.json()["id"]

    async def originate(self, to_number: str) -> None:
        """Initiate an outbound call through Fritzbox. StasisStart fires on answer."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._s.ari_base_url}/ari/channels",
                auth=(self._s.ari_username, self._s.ari_password),
                params={
                    "endpoint": f"PJSIP/{to_number}@fritzbox-endpoint",
                    "callerId": self._s.caller_id,
                    "app": self._s.ari_app_name,
                },
                timeout=10.0,
            )
        resp.raise_for_status()
        log.info("Outbound call to %s initiated", to_number)

    async def _bridge_channels(self, channel_id: str, ext_channel_id: str) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._s.ari_base_url}/ari/bridges",
                auth=(self._s.ari_username, self._s.ari_password),
                params={"type": "mixing", "bridgeId": f"bridge-{channel_id}"},
                timeout=5.0,
            )
            resp.raise_for_status()
            bridge_id = resp.json()["id"]
            await client.post(
                f"{self._s.ari_base_url}/ari/bridges/{bridge_id}/addChannel",
                auth=(self._s.ari_username, self._s.ari_password),
                params={"channel": f"{channel_id},{ext_channel_id}"},
                timeout=5.0,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ari.py -v
```

Expected: all 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add agent/ari.py tests/test_ari.py
git commit -m "feat: ARI client with ExternalMedia bridge and VAD-driven pipeline"
```

---

## Task 13: Main entry point

**Files:**
- Create: `agent/main.py`

No unit test for main — tested end-to-end in Task 15.

- [ ] **Step 1: Create `agent/main.py`**

```python
import asyncio
import logging
import os

import asyncpg
import msal

from agent.config import Settings
from agent.ari import AriClient
from agent.pipeline import VoicePipeline
from agent.stt import SttClient
from agent.tts import TtsClient
from agent.llm import LlmClient
from agent.tools.rag import RagTool
from agent.tools.calendar import MSGraphCalendar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


async def main() -> None:
    s = Settings()

    pg_pool = await asyncpg.create_pool(s.db_dsn, min_size=2, max_size=5)

    msal_app = msal.ConfidentialClientApplication(
        client_id=s.azure_client_id,
        authority=f"https://login.microsoftonline.com/{s.azure_tenant_id}",
        client_credential=s.azure_client_secret,
    )

    stt = SttClient(base_url=s.stt_base_url)
    tts = TtsClient(base_url=s.tts_base_url)
    rag = RagTool(pool=pg_pool, embedding_base_url=s.embedding_base_url)
    calendar = MSGraphCalendar(msal_app=msal_app, user_email=s.calendar_user_email)
    llm = LlmClient(
        base_url=s.llm_base_url,
        model=s.llm_model,
        system_prompt=s.llm_system_prompt,
        rag=rag.lookup,
        calendar=calendar,
    )
    pipeline = VoicePipeline(stt=stt, llm=llm, tts=tts)
    ari = AriClient(settings=s, pipeline=pipeline)

    await ari.run()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Verify import chain works**

```bash
python -c "from agent.main import main; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agent/main.py
git commit -m "feat: main entry point wiring all components"
```

---

## Task 14: Asterisk config

**Files:**
- Create: `asterisk/pjsip.conf`
- Create: `asterisk/extensions.conf`
- Create: `asterisk/ari.conf`
- Create: `asterisk/README.md`

Fill in `<FRITZBOX_IP>`, `<SIP_USER>`, `<SIP_PASSWORD>`, `<YOUR_PHONE_NUMBER>` from Fritzbox web UI (Home Network → Network → IP Addresses for IP, Telephony → Own numbers for SIP credentials).

- [ ] **Step 1: Create `asterisk/pjsip.conf`**

```ini
; ── Transport ──────────────────────────────────────────────────────────────
[transport-udp]
type=transport
protocol=udp
bind=0.0.0.0

; ── Fritzbox SIP trunk ─────────────────────────────────────────────────────
[fritzbox]
type=registration
transport=transport-udp
outbound_auth=fritzbox-auth
server_uri=sip:<FRITZBOX_IP>
client_uri=sip:<SIP_USER>@<FRITZBOX_IP>
retry_interval=30
expiration=120

[fritzbox-auth]
type=auth
auth_type=userpass
username=<SIP_USER>
password=<SIP_PASSWORD>

[fritzbox-aor]
type=aor
contact=sip:<FRITZBOX_IP>

[fritzbox-endpoint]
type=endpoint
transport=transport-udp
context=from-fritzbox
disallow=all
allow=alaw
outbound_auth=fritzbox-auth
aors=fritzbox-aor
from_user=<YOUR_PHONE_NUMBER>
from_domain=<FRITZBOX_IP>

[fritzbox-identify]
type=identify
endpoint=fritzbox-endpoint
match=<FRITZBOX_IP>
```

- [ ] **Step 2: Create `asterisk/extensions.conf`**

```ini
[general]
static=yes
writeprotect=no

; ── Inbound calls from Fritzbox ────────────────────────────────────────────
[from-fritzbox]
exten => _X.,1,NoOp(Inbound call from ${CALLERID(num)})
 same => n,Answer()
 same => n,Stasis(voip-agent)
 same => n,Hangup()

; ── Local test extension (dial 9999 to test pipeline) ─────────────────────
[default]
exten => 9999,1,Answer()
 same => n,Stasis(voip-agent)
 same => n,Hangup()
```

- [ ] **Step 3: Create `asterisk/ari.conf`**

```ini
[general]
enabled=yes
pretty=yes

[voip-agent]
type=user
read_only=no
password=changeme
; Match password in .env ARI_PASSWORD
```

- [ ] **Step 4: Create `asterisk/README.md`**

```markdown
# Asterisk Configuration

Copy these files to `/etc/asterisk/` on the NUC running Asterisk 20.

1. Replace all `<PLACEHOLDER>` values with real Fritzbox credentials.
2. Reload Asterisk: `asterisk -rx "core reload"`
3. Check registration: `asterisk -rx "pjsip show registrations"`
   Expected: `fritzbox` shows `Registered`.
4. Check ARI: `curl -u voip-agent:changeme http://localhost:8088/ari/applications`
   Expected: JSON list including `voip-agent`.
```

- [ ] **Step 5: Commit**

```bash
git add asterisk/
git commit -m "feat: Asterisk PJSIP + ARI config for Fritzbox SIP trunk"
```

---

## Task 15: DGX Docker Compose

**Files:**
- Create: `dgx/docker-compose.yml`
- Create: `dgx/.env.example`
- Create: `dgx/README.md`

- [ ] **Step 1: Create `dgx/docker-compose.yml`**

```yaml
version: "3.9"

services:
  qwen3-asr:
    image: ghcr.io/aeon-7/qwen3-asr-server:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - ASR_MODEL=${ASR_MODEL:-Qwen/Qwen3-ASR-1.7B}
    ports:
      - "8001:8001"
    restart: unless-stopped

  qwen3-tts:
    image: ghcr.io/aeon-7/qwen3-tts-server:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - TTS_MODEL=${TTS_MODEL:-qwen3-tts}
    ports:
      - "8002:8002"
    restart: unless-stopped

  embedding:
    image: ghcr.io/huggingface/text-embeddings-inference:latest
    runtime: nvidia
    command: --model-id intfloat/multilingual-e5-large --port 8003
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "8003:8003"
    restart: unless-stopped
```

- [ ] **Step 2: Create `dgx/.env.example`**

```bash
ASR_MODEL=Qwen/Qwen3-ASR-1.7B
TTS_MODEL=qwen3-tts
```

- [ ] **Step 3: Create `dgx/README.md`**

```markdown
# DGX Spark Services

Run on the DGX Spark (GPU host).

```bash
cd dgx
cp .env.example .env
docker compose up -d
```

## Health checks

```bash
# ASR
curl -s http://localhost:8001/health

# TTS
curl -s http://localhost:8002/health

# Embedding (text-embeddings-inference)
curl -s http://localhost:8003/health
```

Nous Hermes via vLLM is assumed to already be running on its own port.
Set `LLM_BASE_URL` in the agent `.env` to point at it.
```

- [ ] **Step 4: Commit**

```bash
git add dgx/
git commit -m "feat: DGX Docker Compose for Qwen3-ASR, Qwen3-TTS, embedding sidecar"
```

---

## Task 16: End-to-end smoke test

No automated test — requires live Asterisk + DGX services.

- [ ] **Step 1: Start DGX services**

```bash
cd dgx && docker compose up -d
# Wait ~60s for model loads
docker compose logs -f qwen3-asr  # watch for "Uvicorn running"
```

- [ ] **Step 2: Start agent**

```bash
cp .env.example .env
# Fill in real values
python -m agent.main
```

Expected log: `Connecting to ARI at ws://...`

- [ ] **Step 3: Dial test extension from a phone on your LAN**

Dial `9999` from any phone registered on the Fritzbox.

Expected:
1. Agent logs `Call ch-xxx from <your number> ready`
2. You hear the German greeting audio
3. Speak a question in German
4. After ~800ms silence, agent logs `[stt] transcript: ...`
5. You hear the German response within ~3 seconds

- [ ] **Step 4: Check latency breakdown in logs**

```
[stt] transcribed in ~140ms
[llm] responded in ~500ms
[tts] synthesized in ~1500ms
[turn] total ~2200ms
```

If TTS takes >3s: reduce `QWEN_TTS_MODEL` to the 0.6B variant in `dgx/.env`.

- [ ] **Step 5: Commit README with run instructions**

```bash
cat > README.md << 'EOF'
# voip-agent

German-language voice AI agent. Inbound/outbound calls via Fritzbox SIP.

## Prerequisites
- ASUS NUC: Asterisk 20 installed (`apt install asterisk`)
- DGX Spark: Docker with NVIDIA runtime
- Python 3.12 on NUC

## Setup

```bash
# NUC: copy Asterisk config
cp asterisk/*.conf /etc/asterisk/
# Fill placeholders in /etc/asterisk/pjsip.conf
asterisk -rx "core reload"

# DGX: start AI services
cd dgx && docker compose up -d

# NUC: start agent
cp .env.example .env && vim .env
pip install -e .
python -m agent.main
```

## Run tests

```bash
pytest -v
```
EOF
git add README.md
git commit -m "docs: README with setup and run instructions"
```
