# Environment Boundary and Merged-Branch Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the merged Shared-AI branch workspace, correct stale TTS status documentation, and accept the documented Compose-only `DGX_HOST_IP` key without weakening strict agent configuration validation.

**Architecture:** Keep the existing shared `.env` operator workflow and model `DGX_HOST_IP` as an explicit, serialization-excluded `IPvAnyAddress` compatibility field in `Settings`. Preserve `extra_forbidden` for every undeclared key, validate the behavior through temporary dotenv files, and gate branch cleanup on clean/merged provenance checks.

**Tech Stack:** Git worktrees, GitHub CLI, Python 3.14, Pydantic Settings 2.x, pytest, Ruff, Docker Compose.

## Global Constraints

- Remove only `feat/shared-ai-traefik` and `/home/volsch/projekte/.worktrees/voip-agent-shared-ai-traefik`.
- Stop cleanup if the target worktree is dirty or its head is not an ancestor of `main`.
- Do not touch the locked Claude worktree or any unrelated local branch.
- Do not print `.env` values or secrets.
- Keep diagnostic raw TTS streaming and vLLM streamed tool-call fixtures explicitly unverified where live evidence is absent.
- Accept valid IPv4 and IPv6 `DGX_HOST_IP` values.
- Reject invalid `DGX_HOST_IP` values and all other unknown dotenv keys.
- Keep `dgx_host_ip` out of `Settings` representations and serialized model output.

---

### Task 1: Remove the merged Shared-AI worktree and branches

**Files:**
- Remove worktree: `/home/volsch/projekte/.worktrees/voip-agent-shared-ai-traefik`
- No repository files modified

**Interfaces:**
- Consumes: local branch `feat/shared-ai-traefik`, remote branch `origin/feat/shared-ai-traefik`, and base branch `main`
- Produces: no Shared-AI worktree, local branch, or remote branch; all unrelated worktrees remain registered

- [ ] **Step 1: Resolve and verify the exact cleanup target**

Run:

```bash
git -C /home/volsch/projekte/.worktrees/voip-agent-shared-ai-traefik status --short --branch
git -C /home/volsch/projekte/voip-agent merge-base --is-ancestor \
  feat/shared-ai-traefik main
git -C /home/volsch/projekte/voip-agent ls-remote --heads \
  origin feat/shared-ai-traefik
git -C /home/volsch/projekte/voip-agent worktree list --porcelain
```

Expected:

- the target worktree has no changed or untracked files;
- `merge-base --is-ancestor` exits 0;
- the remote branch resolves to the same target head;
- the target path is under `/home/volsch/projekte/.worktrees/`;
- the locked Claude worktree is visible but remains outside the cleanup target.

- [ ] **Step 2: Delete only the merged remote branch**

Run:

```bash
git -C /home/volsch/projekte/voip-agent push \
  origin --delete feat/shared-ai-traefik
```

Expected: Git reports deletion of `feat/shared-ai-traefik` on `origin`.

- [ ] **Step 3: Remove the owned worktree and local branch**

Run:

```bash
git -C /home/volsch/projekte/voip-agent worktree remove \
  /home/volsch/projekte/.worktrees/voip-agent-shared-ai-traefik
git -C /home/volsch/projekte/voip-agent worktree prune
git -C /home/volsch/projekte/voip-agent branch -d feat/shared-ai-traefik
```

Expected: the worktree is removed and Git deletes the fully merged local
branch without force.

- [ ] **Step 4: Verify cleanup without touching unrelated worktrees**

Run:

```bash
git -C /home/volsch/projekte/voip-agent worktree list --porcelain
git -C /home/volsch/projekte/voip-agent branch --list feat/shared-ai-traefik
git -C /home/volsch/projekte/voip-agent ls-remote --heads \
  origin feat/shared-ai-traefik
```

Expected: all three target lookups are empty while the locked Claude worktree
remains registered.

### Task 2: Align the deferred TTS documentation with current evidence

**Files:**
- Modify: `TODO.md:56-66`

**Interfaces:**
- Consumes: the deployed stable named clone profile, whole-WAV endpoint,
  cooperative cancellation, and GX10 live verification evidence
- Produces: an accurate status entry that separates production acceptance from
  diagnostic/raw-stream and vLLM fixture assumptions

- [ ] **Step 1: Replace the stale faster-qwen3-tts status**

Replace `TODO.md` lines 56-66 with:

```markdown
- [x] **faster-qwen3-tts (stable named clone production path).** Production
  uses the private `shared-female-de-v1` profile with the pinned Qwen Base
  clone runtime. Each response sentence uses stable `/v1/audio/speech`
  whole-WAV synthesis at 24 kHz PCM16; diagnostic
  `/v1/audio/speech/stream` remains outside the VoIP response path.
  Cooperative cancellation now stops active synthesis between codec steps,
  synchronizes CUDA before releasing the exclusive model lock, and lets the
  following request proceed without waiting for the cancelled sentence.
  GX10 verification on 2026-07-30 covered the image-build patch verifier,
  healthy zero-restart deployment, authenticated stable WAV synthesis,
  cancellation/recovery, and real GPU activity.
  Raw diagnostic-stream wire shape and vLLM streamed `delta.tool_calls`
  fragments remain unit-fixture assumptions; verify those paths live before
  treating them as production-proven contracts.
```

- [ ] **Step 2: Check documentation formatting and scope**

Run:

```bash
git diff --check
git diff -- TODO.md
rg -n "DGX unreachable|VoiceDesign|ASSUMPTION|delta\\.tool_calls" TODO.md
```

Expected:

- `git diff --check` exits 0;
- the stale “DGX unreachable” text is absent;
- the updated entry does not claim raw diagnostic streaming or vLLM tool-call
  fragmentation was live-verified.

- [ ] **Step 3: Commit the documentation correction**

Run:

```bash
git add TODO.md
git commit -m "docs(tts): align deferred verification status"
```

Expected: one documentation-only commit containing only `TODO.md`.

### Task 3: Accept the documented Compose host without weakening strict validation

**Files:**
- Modify: `tests/test_config.py`
- Modify: `agent/config.py:3-15`

**Interfaces:**
- Consumes: dotenv key `DGX_HOST_IP`
- Produces: `Settings.dgx_host_ip: IPvAnyAddress`, excluded from `repr()` and
  `model_dump()`

- [ ] **Step 1: Write the failing valid-dotenv regression test**

Add to `tests/test_config.py`:

```python
@pytest.mark.parametrize("value", ["192.168.68.41", "fd00::41"])
def test_settings_accept_documented_compose_host(tmp_path, value: str) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"DGX_HOST_IP={value}\n", encoding="utf-8")

    settings = Settings(_env_file=env_file, **_valid_kwargs())

    assert str(settings.dgx_host_ip) == value
    assert "dgx_host_ip" not in repr(settings)
    assert "dgx_host_ip" not in settings.model_dump()
```

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```bash
venv/bin/pytest \
  tests/test_config.py::test_settings_accept_documented_compose_host \
  -v --tb=short
```

Expected: both parameter cases fail because `DGX_HOST_IP` is currently rejected
as `extra_forbidden`.

- [ ] **Step 3: Add the minimal typed compatibility field**

Change the imports at the top of `agent/config.py` to:

```python
from pydantic import ConfigDict, Field, IPvAnyAddress, field_validator, model_validator
```

Add directly below `model_config`:

```python
    # Compose reads this documented key from the same .env file. The agent does
    # not consume it, but declaring and validating it preserves strict rejection
    # of every other unknown setting.
    dgx_host_ip: IPvAnyAddress = Field(
        default="192.168.68.41",
        exclude=True,
        repr=False,
        validate_default=True,
    )
```

- [ ] **Step 4: Run the valid-dotenv regression test and verify GREEN**

Run:

```bash
venv/bin/pytest \
  tests/test_config.py::test_settings_accept_documented_compose_host \
  -v --tb=short
```

Expected: both IPv4 and IPv6 cases pass.

- [ ] **Step 5: Add preservation tests for invalid and unrelated keys**

Add to `tests/test_config.py`:

```python
def test_settings_reject_invalid_compose_host(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("DGX_HOST_IP=not-an-ip\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dgx_host_ip"):
        Settings(_env_file=env_file, **_valid_kwargs())


def test_settings_still_reject_unknown_dotenv_keys(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MISSPELLED_SETTING=value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="extra_forbidden"):
        Settings(_env_file=env_file, **_valid_kwargs())
```

- [ ] **Step 6: Run the complete configuration suite**

Run:

```bash
venv/bin/pytest tests/test_config.py -v --tb=short
```

Expected: all configuration tests pass.

- [ ] **Step 7: Verify the real repository dotenv boundary without exposing values**

Run:

```bash
venv/bin/python -c 'from agent.config import Settings; s = Settings(); print(type(s.dgx_host_ip).__name__, "dgx_host_ip" in s.model_dump())'
```

Expected: output identifies an IPv4 or IPv6 address type followed by `False`;
no dotenv value is printed.

- [ ] **Step 8: Commit the configuration fix**

Run:

```bash
git add agent/config.py tests/test_config.py
git commit -m "fix(config): accept documented compose host"
```

Expected: one implementation commit containing only the settings field and its
regression tests.

### Task 4: Run full verification and report the final state

**Files:**
- Verify: repository worktree and remote state
- Verify: `agent/`, `tests/`, `dgx/tts/`, `compose.yml`,
  `compose.pjsip-poc.yml`, `dgx/docker-compose.yml`

**Interfaces:**
- Consumes: completed cleanup, documentation update, and configuration fix
- Produces: fresh acceptance evidence for tests, lint, formatting, Compose,
  dotenv loading, branch cleanup, and repository status

- [ ] **Step 1: Run the complete unit suite**

Run from `/tmp` so the production `.env` cannot influence unit fixtures:

```bash
/home/volsch/projekte/voip-agent/venv/bin/pytest \
  -q \
  --rootdir=/home/volsch/projekte/voip-agent \
  /home/volsch/projekte/voip-agent/tests
```

Expected: all tests pass with zero failures or errors.

- [ ] **Step 2: Run Ruff and diff checks**

Run:

```bash
venv/bin/ruff check agent tests dgx/tts
venv/bin/ruff format --check agent tests dgx/tts
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 3: Validate all Compose configurations**

Run:

```bash
docker compose -p voip-agent \
  --project-directory /home/volsch/projekte/voip-agent \
  -f /home/volsch/projekte/voip-agent/compose.yml config --quiet
docker compose -p voip-agent \
  --project-directory /home/volsch/projekte/voip-agent \
  -f /home/volsch/projekte/voip-agent/compose.pjsip-poc.yml config --quiet
docker compose \
  --project-directory /home/volsch/projekte/voip-agent/dgx \
  -f /home/volsch/projekte/voip-agent/dgx/docker-compose.yml config --quiet
```

Expected: all three commands exit 0.

- [ ] **Step 4: Verify cleanup, commits, and repository status**

Run:

```bash
git worktree list --porcelain
git branch --list feat/shared-ai-traefik
git ls-remote --heads origin feat/shared-ai-traefik
git status --short --branch
git log -4 --oneline
```

Expected:

- no Shared-AI worktree or branch exists locally or remotely;
- unrelated worktrees remain registered;
- `main` contains the design, plan, documentation, and configuration commits;
- the working tree is clean.
