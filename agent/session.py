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
    (SessionState.PROCESSING, SessionState.LISTENING),
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
