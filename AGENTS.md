# Repository Guidelines

## Project Structure & Module Organization

`agent/` contains the Python 3.12+ asyncio application. Core call orchestration lives in `ari.py`, `pipeline.py`, and `session.py`; media handling is in `audio.py`, `rtp.py`, `stt.py`, and `tts.py`; integrations are under `agent/tools/`. Keep configuration centralized in `agent/config.py`. Tests mirror these modules in `tests/test_*.py`, with shared fixtures in `tests/conftest.py`. Asterisk templates live in `asterisk/`, DGX-hosted inference services and Compose files in `dgx/`, and design records in `docs/`.

## Build, Test, and Development Commands

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"          # install package and test dependencies
pytest -v --tb=short             # run the complete unit suite
pytest tests/test_pipeline.py -v # run one module
ruff check agent/ tests/         # lint imports and Python errors
ruff format --check agent/ tests/ # verify formatting
python -m agent.main             # start the configured agent locally
cd dgx && docker compose up -d   # start DGX inference services
```

## Coding Style & Naming Conventions

Use four-space indentation, type hints, and focused async functions. Ruff enforces `E`, `F`, and `I` rules, a 100-character line limit, and formatting. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and leading underscores for internal helpers. Preserve the session state machine and keep blocking SDK or model work off the event loop with `asyncio.to_thread`.

## Testing Guidelines

Pytest uses `asyncio_mode = "auto"`, so async tests are plain `async def` functions. Name tests `test_<behavior>` and place them beside the corresponding module-level suite. Mock HTTP with `respx` and async collaborators with `AsyncMock`; unit tests must not contact real AI services, databases, or Microsoft Graph. Add regression tests for state transitions, cancellation/barge-in, malformed RTP, and failure fallbacks. CI runs tests on Python 3.12 and 3.13.

## Commit & Pull Request Guidelines

Follow the history’s Conventional Commit pattern: `feat: ...`, `fix(ari,llm): ...`, `docs(tts): ...`, or `style: ...`. Keep subjects imperative and scoped; reference issues or PRs when applicable. Pull requests should explain user-visible behavior, architecture or configuration changes, tests run, and any live Asterisk/DGX validation still required. Update README or design docs when wire contracts, environment variables, or deployment steps change.

## Security & Configuration

Copy `.env.example` to `.env`; never commit secrets or real Fritzbox, ARI, Azure, caller, or database credentials. Keep tool access fail-closed through `TRUSTED_CALLERS`, and explicitly enable calendar writes only where intended. Treat `asterisk/*.conf` as templates and replace placeholders only in deployment copies.
