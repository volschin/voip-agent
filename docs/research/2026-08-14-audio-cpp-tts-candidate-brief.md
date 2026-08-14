# Candidate Brief: audio.cpp Qwen3-TTS runtime

**Datum:** 2026-08-14 UTC
**Status:** Statische Upstream-Gates eingefroren; Runtime nur nach deren PASS
**Maximal zulässiges Ergebnis:** `eligible`
**Nicht freigegeben:** Empfehlung, Adoption oder Produktions-Rollout

## Claim und Identität

Geprüft wird, ob `audio.cpp` die produktive Qwen3-TTS-Base-Stimme lokal und
vertragstreu ersetzen könnte. Das Modell bleibt identisch; Runtime und Server
wechseln.

- Runtime-Source, Image und Digests: identisch zum ASR-Brief, Revision
  `04ba4375ed53bbd718bd2697e190007f6a19f426`
- Modell: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, Snapshot
  `fd4b254389122332181a7c3db7f27e918eec64e3`
- Voice-ID: `shared-female-de-v1`
- private Referenz-WAV: 457.004 Bytes, SHA-256
  `707085e41a951c8ad7a6fec3e101a4ffdc40a47192f53a0fe8d1fc63eb807ece`
- privater Referenztext: nur als SHA-256
  `f46801ea847c972f12cdba959c701d4ea516c3be1ce67b2f816c41cf2d7f8fdf`
- Servermodell: ID `qwen3-tts`, Family `qwen3_tts`, Task `tts`, Mode
  `offline`, Backend `cuda`, Voice-Preset exakt auf obige Referenz

Private Stimme und Text bleiben owner-only, read-only und werden weder in Logs
noch in Git oder externe Dienste übertragen.

## Harte Reihenfolge

Vor einem Containerstart muss die unveränderte Upstream-Implementierung
nachweisen, dass ein vom `voip-agent` abgebrochener HTTP-Request die laufende
Synthese kooperativ beendet, den exklusiven Modell-Lock freigibt und keine
vollständige verworfene WAV weiterberechnet. Das ist ein produktives
Latenz-/Barge-in-Invariant, kein Optimierungswunsch.

Nur bei statischem PASS folgen: exakter Consumer-Vertrag (`input`, `voice`,
`language`, ohne `model`) über einen explizit eingefrorenen Adapter; ganze
24-kHz-Mono-PCM16-WAV; kein Streamingpfad; stabile Voice-ID; deutsche
Roundtrip-Verständlichkeit und kritische Entitäten; Latenz und Concurrency;
Abbruch unter Last; lokale/offline Ausführung und PID-korreliertes CUDA.

Der produktive TTS-Client hat SHA-256
`5babbd041f2869fef85f3d37ce5b73a9ea2e7bd6a79383abbb9fd4597fc7669b`.
Die bestehende Produktionsimplementierung besitzt einen expliziten
kooperativen Cancel-Pfad; ein Busy-Timeout, das nur Folgerequests mit 503
abweist, ist kein Ersatz.

Ein fehlendes Hard Gate beendet den Claim als `ineligible`; nach diesem
Fail-fast-Punkt werden keine Qualitäts- oder Lastmessungen gestartet. Eine
Abweichung der bezeichneten Artefakte oder eine nachträgliche Reparatur macht
den Lauf `invalid`. Genau wie beim ASR-Claim sind Empfehlung, Adoption und
Rollout außerhalb der Autorität.
