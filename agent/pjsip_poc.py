"""Headless PJSUA2 proof of concept for delayed FRITZ!Box call answering.

This module deliberately stops at SIP signalling. It registers as a FRITZ!Box
IP telephone, lets the regular phones ring, and accepts an unanswered call
after a configurable delay. It exists only as a signalling diagnostic; the
production transport is ``agent.pjsip``. Both share the answer policy in
``agent.answer_policy``.
"""

from __future__ import annotations

import importlib
import logging
import os
import signal
from dataclasses import dataclass

from agent.answer_policy import DelayedAnswerService, caller_id_from_uri

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _env_float(name: str, default: float, minimum: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class PjsipPocSettings:
    """Configuration needed by the isolated PJSIP proof of concept."""

    fritzbox_host: str
    sip_username: str
    sip_password: str
    sip_transport: str = "udp"
    sip_local_port: int = 5062
    answer_delay_seconds: float = 20.0
    max_call_seconds: float = 30.0
    max_concurrent_calls: int = 1
    pjsip_log_level: int = 4
    event_poll_ms: int = 50

    @classmethod
    def from_env(cls) -> PjsipPocSettings:
        host = os.getenv("FRITZBOX_HOST", "fritz.box").strip()
        username = os.getenv("FRITZBOX_SIP_USERNAME", "").strip()
        password = os.getenv("FRITZBOX_SIP_PASSWORD", "").strip()
        transport = os.getenv("PJSIP_TRANSPORT", "udp").strip().lower()

        if not host:
            raise ValueError("FRITZBOX_HOST must not be empty")
        if not username:
            raise ValueError("FRITZBOX_SIP_USERNAME must not be empty")
        if not password or password.lower() == "changeme":
            raise ValueError("FRITZBOX_SIP_PASSWORD must be a non-placeholder secret")
        if transport not in {"udp", "tcp"}:
            raise ValueError("PJSIP_TRANSPORT must be 'udp' or 'tcp'")

        return cls(
            fritzbox_host=host,
            sip_username=username,
            sip_password=password,
            sip_transport=transport,
            sip_local_port=_env_int("PJSIP_LOCAL_PORT", 5062, 1024, 65535),
            answer_delay_seconds=_env_float("ANSWER_DELAY_SECONDS", 20.0, 1.0),
            max_call_seconds=_env_float("POC_MAX_CALL_SECONDS", 30.0, 1.0),
            max_concurrent_calls=_env_int("MAX_CONCURRENT_CALLS", 1, 1, 32),
            pjsip_log_level=_env_int("PJSIP_LOG_LEVEL", 4, 0, 6),
            event_poll_ms=_env_int("PJSIP_EVENT_POLL_MS", 50, 10, 1000),
        )

    @property
    def registrar_uri(self) -> str:
        uri = f"sip:{self.fritzbox_host}"
        if self.sip_transport == "tcp":
            return f"{uri};transport=tcp"
        return uri

    @property
    def identity_uri(self) -> str:
        return f"sip:{self.sip_username}@{self.fritzbox_host}"


class PjsipPhone:
    """Own the PJSUA2 endpoint and pump it on the process main thread."""

    def __init__(self, settings: PjsipPocSettings) -> None:
        self._settings = settings
        self._stop_requested = False
        self._endpoint = None
        self._account = None
        self._service = DelayedAnswerService(
            answer_delay_seconds=settings.answer_delay_seconds,
            max_call_seconds=settings.max_call_seconds,
            max_concurrent_calls=settings.max_concurrent_calls,
        )

    def request_stop(self, *_args: object) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            pj = importlib.import_module("pjsua2")
        except ImportError as exc:
            raise RuntimeError(
                "pjsua2 is unavailable; run this entry point through Dockerfile.pjsip-poc"
            ) from exc

        phone = self

        class PocCall(pj.Call):
            def __init__(self, account: object, native_call_id: int) -> None:
                super().__init__(account, native_call_id)
                self.native_call_id = native_call_id

            def _reply(self, status_code: int) -> None:
                prm = pj.CallOpParam()
                prm.statusCode = status_code
                super().answer(prm)

            def signal_ringing(self) -> None:
                self._reply(pj.PJSIP_SC_RINGING)

            def accept(self) -> None:
                self._reply(pj.PJSIP_SC_OK)

            def reject_busy(self) -> None:
                self._reply(pj.PJSIP_SC_BUSY_HERE)

            def terminate(self) -> None:
                prm = pj.CallOpParam()
                prm.statusCode = pj.PJSIP_SC_DECLINE
                super().hangup(prm)

            def onCallState(self, _prm: object) -> None:  # noqa: N802 - PJSUA2 callback
                try:
                    info = self.getInfo()
                except Exception:
                    logger.exception("Could not query state for call %s", self.native_call_id)
                    return
                logger.info(
                    "Call %s state=%s status=%s %s",
                    self.native_call_id,
                    info.stateText,
                    info.lastStatusCode,
                    info.lastReason,
                )
                if info.state == pj.PJSIP_INV_STATE_DISCONNECTED:
                    phone._service.disconnected(
                        self.native_call_id,
                        int(info.lastStatusCode),
                        info.lastReason,
                    )
                    phone._account.defer_cleanup(self.native_call_id)

            def onCallMediaState(self, _prm: object) -> None:  # noqa: N802
                logger.info(
                    "Call %s media is active (audio bridge intentionally pending)",
                    self.native_call_id,
                )

        class PocAccount(pj.Account):
            def __init__(self) -> None:
                super().__init__()
                self.calls: dict[int, PocCall] = {}
                self.cleanup_ids: set[int] = set()

            def onRegState(self, prm: object) -> None:  # noqa: N802 - PJSUA2 callback
                info = self.getInfo()
                logger.info(
                    "SIP registration active=%s status=%s %s",
                    info.regIsActive,
                    prm.code,
                    prm.reason,
                )

            def onIncomingCall(self, prm: object) -> None:  # noqa: N802 - PJSUA2 callback
                call = PocCall(self, prm.callId)
                self.calls[prm.callId] = call
                try:
                    caller = caller_id_from_uri(call.getInfo().remoteUri)
                    phone._service.offer(prm.callId, caller, call)
                except Exception:
                    logger.exception("Failed to process incoming call %s", prm.callId)
                    try:
                        call.reject_busy()
                    except Exception:
                        logger.exception("Failed to reject broken call %s", prm.callId)

            def defer_cleanup(self, call_id: int) -> None:
                self.cleanup_ids.add(call_id)

            def cleanup(self) -> None:
                for call_id in self.cleanup_ids:
                    self.calls.pop(call_id, None)
                self.cleanup_ids.clear()

        endpoint = pj.Endpoint()
        self._endpoint = endpoint
        endpoint.libCreate()

        endpoint_config = pj.EpConfig()
        endpoint_config.uaConfig.threadCnt = 0
        endpoint_config.uaConfig.mainThreadOnly = True
        endpoint_config.uaConfig.userAgent = "voip-agent-pjsip-poc/0.1"
        endpoint_config.logConfig.level = self._settings.pjsip_log_level
        endpoint_config.logConfig.consoleLevel = self._settings.pjsip_log_level
        endpoint.libInit(endpoint_config)

        transport_config = pj.TransportConfig()
        transport_config.port = self._settings.sip_local_port
        transport_type = {
            "udp": pj.PJSIP_TRANSPORT_UDP,
            "tcp": pj.PJSIP_TRANSPORT_TCP,
        }[self._settings.sip_transport]
        transport_id = endpoint.transportCreate(transport_type, transport_config)
        endpoint.libStart()
        endpoint.audDevManager().setNullDev()

        account_config = pj.AccountConfig()
        account_config.idUri = self._settings.identity_uri
        account_config.regConfig.registrarUri = self._settings.registrar_uri
        account_config.sipConfig.transportId = transport_id
        credential = pj.AuthCredInfo(
            "digest",
            "*",
            self._settings.sip_username,
            0,
            self._settings.sip_password,
        )
        account_config.sipConfig.authCreds.append(credential)

        account = PocAccount()
        self._account = account
        account.create(account_config)

        logger.info(
            "PJSIP PoC started: registrar=%s transport=%s local_port=%s",
            self._settings.registrar_uri,
            self._settings.sip_transport,
            self._settings.sip_local_port,
        )

        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

        try:
            while not self._stop_requested:
                endpoint.libHandleEvents(self._settings.event_poll_ms)
                self._service.tick()
                account.cleanup()
        finally:
            logger.info("Shutting down PJSIP PoC")
            self._service.terminate_all()
            account.cleanup()
            account.shutdown()
            self._account = None
            endpoint.libDestroy()
            self._endpoint = None


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    phone = PjsipPhone(PjsipPocSettings.from_env())
    phone.run()


if __name__ == "__main__":
    main()
