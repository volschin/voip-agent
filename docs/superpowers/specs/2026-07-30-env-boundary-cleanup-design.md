# Environment Boundary and Merged-Branch Cleanup Design

**Date:** 2026-07-30

## Goal

Remove the already-merged `feat/shared-ai-traefik` worktree and branches, align
`TODO.md` with the verified production TTS architecture, and let the documented
Compose-only `DGX_HOST_IP` entry coexist with strict agent configuration
validation.

## Scope

- Remove only the clean `feat/shared-ai-traefik` worktree, its local branch, and
  its remote branch after re-confirming that its head is an ancestor of
  `main`.
- Update the stale faster-qwen3-tts section in `TODO.md` to describe the named
  Qwen Base clone profile, stable whole-WAV production endpoint, cooperative
  cancellation, and completed GX10 verification.
- Keep diagnostic raw streaming and vLLM streamed tool-call fixture assumptions
  explicitly non-production and unverified where current evidence does not
  prove them.
- Accept `DGX_HOST_IP` as an explicit, typed compatibility field in
  `agent.config.Settings`.
- Preserve strict rejection of every other unknown `.env` key.

## Configuration Design

The repository intentionally shares `.env` between Docker Compose interpolation
and local agent configuration. `.env.example` and `compose.yml` document
`DGX_HOST_IP`, but `Settings` currently reads the same file and rejects that
Compose-only key as an extra field.

`Settings` will declare `dgx_host_ip` as a validated IP-address field. The agent
does not consume this value at runtime; declaring it records the shared-file
contract and validates the value before local startup. The field will be hidden
from representations and excluded from model serialization so it does not
become part of the agent's runtime configuration surface.

This is preferred over `extra="ignore"` because unknown variables and spelling
mistakes must continue to fail closed. Splitting the files is intentionally out
of scope because it would change the established operator workflow and require
migration of existing private deployment configuration.

## Failure Behavior

- A valid IPv4 or IPv6 `DGX_HOST_IP` is accepted.
- An invalid `DGX_HOST_IP` fails settings validation.
- Any other unknown `.env` key continues to fail with `extra_forbidden`.
- Cleanup stops before deletion if the target worktree is dirty or its branch
  is not already contained in `main`.

## Verification

The configuration change follows red-green TDD:

1. Add regression tests that load temporary `.env` files.
2. Confirm the documented valid `DGX_HOST_IP` case fails before implementation.
3. Add the minimal typed compatibility field.
4. Confirm valid IPv4/IPv6 values pass, an invalid value fails, and an unrelated
   unknown variable remains rejected.

Final verification includes the complete pytest suite, Ruff check and format
check, all three Compose configuration checks, a direct `Settings()` load from
the real repository `.env` without revealing values, clean Git status apart
from the intended changes, and proof that the removed worktree and branch no
longer exist locally or remotely.
