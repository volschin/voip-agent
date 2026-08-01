"""Delayed-answer and call-duration policy for incoming FRITZ!Box calls.

The policy is transport-neutral: it only talks to the small ``CallActions``
boundary, so it is driven by the production PJSUA2 client in ``agent.pjsip``
and by the signalling-only proof of concept in ``agent.pjsip_poc`` alike.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

logger = logging.getLogger(__name__)

_CALLER_URI = re.compile(r"sip:([^@;>]+)", re.IGNORECASE)

_UNKNOWN_CALLER = "unknown"


def caller_id_from_uri(remote_uri: str) -> str:
    """Extract the SIP user while keeping the raw URI out of application logs."""

    match = _CALLER_URI.search(remote_uri)
    return match.group(1) if match else ""


# Separators an operator may paste from a contact card. Stripped before the
# number is classified; they carry no dialling meaning.
_SEPARATORS = re.compile(r"[\s\-/().]")
# German-only agent, and the trunk is a German FRITZ!Box, so the national prefix
# `0` can only mean +49. Do not generalise this without knowing the trunk's
# country: mapping `0` to the wrong country code would silently authorize a
# different subscriber.
_COUNTRY_CODE = "+49"


def normalize_caller_id(caller: str) -> str:
    """Map one phone number onto a single comparison key.

    The FRITZ!Box delivers external callers in national format
    (``015100000001``) while operators naturally write E.164
    (``+4915100000001``) in ``TRUSTED_CALLERS``; an exact string match failed
    closed with nothing to indicate why. Both sides run through this function
    so the two spellings meet.

    Only unambiguous dialling-plan transforms are applied. Anything that is not
    a plain number after separator removal — internal extensions (``**613``),
    a withheld CLI (``anonymous``) — is returned stripped but otherwise
    untouched, so it stays an exact match and cannot collide with a real
    number. This is an authorization boundary: widening it is a security bug.
    """

    stripped = _SEPARATORS.sub("", caller.strip())
    if not stripped:
        return ""
    if stripped.startswith("+"):
        rest = stripped[1:]
        return stripped if rest.isdigit() else caller.strip()
    if not stripped.isdigit():
        # Extensions and non-numeric CLI: exact-match domain.
        return caller.strip()
    if stripped.startswith("00"):
        return "+" + stripped[2:]
    if stripped.startswith("0"):
        return _COUNTRY_CODE + stripped[1:]
    # Bare digits with no trunk or country prefix carry no country context
    # (a short code, or an extension). Leave them alone rather than guessing.
    return stripped


class CallActions(Protocol):
    """Small boundary between the testable policy and native PJSUA2 calls."""

    def signal_ringing(self) -> None: ...

    def accept(self) -> None: ...

    def reject_busy(self) -> None: ...

    def terminate(self) -> None: ...


class PendingState(Enum):
    WAITING = "waiting"
    ANSWERING = "answering"
    ACTIVE = "active"


@dataclass(slots=True)
class PendingCall:
    call_id: int
    caller: str
    actions: CallActions
    answer_at: float
    state: PendingState = PendingState.WAITING
    active_at: float | None = None


class DelayedAnswerService:
    """Apply delayed-answer and maximum-duration policy to incoming calls."""

    def __init__(
        self,
        *,
        answer_delay_seconds: float,
        max_call_seconds: float,
        max_concurrent_calls: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._answer_delay = answer_delay_seconds
        self._max_call_seconds = max_call_seconds
        self._max_concurrent_calls = max_concurrent_calls
        self._clock = clock
        self._calls: dict[int, PendingCall] = {}

    @property
    def call_count(self) -> int:
        return len(self._calls)

    def offer(self, call_id: int, caller: str, actions: CallActions) -> bool:
        caller = caller or _UNKNOWN_CALLER
        if call_id in self._calls:
            raise ValueError(f"call {call_id} is already tracked")
        if len(self._calls) >= self._max_concurrent_calls:
            logger.warning(
                "Rejecting call %s from %s: capacity reached",
                call_id,
                caller,
            )
            actions.reject_busy()
            return False

        now = self._clock()
        actions.signal_ringing()
        self._calls[call_id] = PendingCall(
            call_id=call_id,
            caller=caller,
            actions=actions,
            answer_at=now + self._answer_delay,
        )
        logger.info(
            "Call %s from %s waiting %.1f seconds before answer",
            call_id,
            caller,
            self._answer_delay,
        )
        return True

    def disconnected(self, call_id: int, status_code: int, reason: str) -> None:
        pending = self._calls.pop(call_id, None)
        if pending is None:
            return
        if pending.state is PendingState.WAITING:
            logger.info(
                "Call %s from %s cancelled before agent answer (status=%s %s)",
                call_id,
                pending.caller,
                status_code,
                reason,
            )
        else:
            logger.info(
                "Call %s from %s ended (status=%s %s)",
                call_id,
                pending.caller,
                status_code,
                reason,
            )

    def tick(self) -> None:
        now = self._clock()
        for call_id, pending in list(self._calls.items()):
            if pending.state is PendingState.WAITING and now >= pending.answer_at:
                pending.state = PendingState.ANSWERING
                try:
                    pending.actions.accept()
                except Exception:
                    self._calls.pop(call_id, None)
                    logger.exception("Failed to answer call %s", call_id)
                    continue
                if self._calls.get(call_id) is pending:
                    pending.state = PendingState.ACTIVE
                    pending.active_at = now
                    logger.info(
                        "Call %s from %s answered by agent",
                        call_id,
                        pending.caller,
                    )
                continue

            if (
                pending.state is PendingState.ACTIVE
                and pending.active_at is not None
                and now - pending.active_at >= self._max_call_seconds
            ):
                logger.info(
                    "Call %s from %s reached the %.0f second duration limit; hanging up",
                    call_id,
                    pending.caller,
                    self._max_call_seconds,
                )
                try:
                    pending.actions.terminate()
                except Exception:
                    logger.exception("Failed to terminate expired call %s", call_id)
                self._calls.pop(call_id, None)

    def terminate_all(self) -> None:
        for pending in list(self._calls.values()):
            try:
                pending.actions.terminate()
            except Exception:
                logger.exception("Failed to terminate call %s during shutdown", pending.call_id)
        self._calls.clear()
