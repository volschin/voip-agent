# Candidate Brief: audio.cpp 0.6-era Qwen3-ASR deployment

**Datum:** 2026-08-14 UTC
**Status:** Für genau einen kontrollierten, isolierten Qualifikationslauf eingefroren
**Maximal zulässiges Ergebnis:** `eligible`
**Nicht freigegeben:** Empfehlung, Adoption oder Produktions-Rollout

## Claim und Kandidatenidentität

Geprüft wird, ob die unten vollständig bezeichnete `audio.cpp`-Deployment-
Identität den gemeinsamen lokalen deutschen ASR-Vertrag von `voip-agent` und
`ai-companion` mindestens so gut wie die aktuelle Produktion erfüllt. Der
Kandidat ist ausdrücklich **kein Runtime-only-Vergleich**: Gegenüber der
Produktion wechseln Runtime, Modelllayout und Präzision gemeinsam.

- Runtime-Repository: `0xShug0/audio.cpp`, Apache-2.0
- Source-Revision: `04ba4375ed53bbd718bd2697e190007f6a19f426`
- Image-Tag: `ghcr.io/0xshug0/audio.cpp:full-cuda13-20260814-04ba437`
- OCI-Index: `sha256:6d34bf5008c840e03fa279d45b9c02c69308ac7430d2be60d517db4b6362ca0e`
- Linux/arm64-Manifest: `sha256:6cddc4a8c306fe07e7baa6d4562d7145777e8eb2664c7c6a45091de67f9881ba`
- Model: `Qwen/Qwen3-ASR-1.7B-hf`, Apache-2.0, BF16/safetensors
- Model-Revision: `bcd2b5b7f32b480ab5790554cfa8347f246a14f3`
- `model.safetensors`: 4.076.193.080 Bytes; upstream LFS SHA-256
  `2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1`
- Servermodell: ID `qwen3-asr`, Family `qwen3_asr`, Task `asr`, Mode
  `offline`, Backend `cuda`, Device `0`, Threads `1`, eager load
- feste Request-Sprache: `de`; keine automatische Spracherkennung

Der aktuelle Produktionsvergleich bleibt
`UrocyonF/Qwen3-ASR-1.7B-NVFP4@61ad4d533c64e033a750b66c44aad6f18634997e`
im Image
`sha256:0fadf01c8957a91ad83aca03395e7cd61fb66c1b20f5049e268ddd5424560930`.

## Erlaubtes Delta und Consumer-Vertrag

Die echten Consumer senden `POST /v1/audio/transcriptions` als Multipart mit
`file` und `language=de`, erwarten `data["text"]` und senden kein `model`.
`audio.cpp` verlangt `model` und liefert zusätzlich `timing`. Der Kandidat
enthält deshalb einen minimalen Qualifikations-Gateway, der ausschließlich
`model=qwen3-asr` ergänzt und die Antwort auf `{text}` normalisiert. Externer
Request, Queueing, Limits und Scoring bleiben unverändert.

- `voip-agent` Consumer `agent/stt.py`:
  `679ffa6342dcf1fb94ed011c89cd6de353c488dab8523feaf18c840205fdbc8c`
- Gateway-Basis `client.py`:
  `2fc7a25ca0be888a6ad60bb5094a94cb133907d6c7cfbdcd5c9073b0001851de`
- Gateway-Basis `app.py`:
  `e3206982f87074bfa941237894c81862743e11590a42640117b33e42143689be`
- einziger Overlay-Patch:
  `2026-08-14-audio-cpp-asr-gateway.patch`, SHA-256
  `96f3107904e1db2a706b64e78f1c96f65a4d0087edb052dc20e7a19d1599cd9d`

Jede weitere Code-, Modell-, Image-, Prompt-, Sprach-, Sampling- oder
Parameteränderung ist verboten und macht die Serie `invalid`.

## Provenienz-, Datenschutz- und Start-Gates

Der Kandidat wird nur in einem owner-only Laufverzeichnis betrieben. Modell,
Config und Belege werden vor Inferenz vollständig gehasht und danach read-only
gemountet. Image-Digest, OCI-Revision-Label, Plattform, Entry Point und der
exakte Upstream-Commit müssen übereinstimmen. Audio und Transkripte bleiben
lokal; der Kandidat erhält kein externes Netz und keine Secrets. Der Host-Port
des Gateways wird ausschließlich an `127.0.0.1` gebunden.

Vor der Messung müssen `/health`, `/v1/models`, der externe Consumer-Vertrag,
eager Model Load, PID-korrelierte CUDA-Ausführung und der unveränderte
Gateway-Hash bestanden sein. Fallback, CPU-Ausführung, nachträglicher Download,
Modellsubstitution oder Reparatur sind `invalid`.

## Eingefrorener Messpfad

- Corpus: `asr-companion-de-v1+4519e93d3f8f6b99`, Evidenzgrad
  `synthetic_bootstrap`, 70 Fälle, maximal etwa 5,2 Sekunden
- versiegelte Referenzen:
  `1f3add58639ac89d35f8112b29cdef3c621058803d8268be3a558a385e253831`
- Benchmark-Runner:
  `091c57013d1422c09bacf72060b95a593a91a8171661d4f6ebe80d29c218cdb1`
- Scorer:
  `e4d655c39e56ff9c09bd7362753b9c3770a0ef29e2281171d8aacb6331c41bf2`
- Corpus-Validator:
  `1f1279e8c5d836a885aa27632364ceec36615a40443ae83484fc5e1d7c6876a1`
- etablierter Non-Speech-Harness, Basis:
  `288127010f43c8df3e998855eee5754b37f56778c8fbc0d72fad7030776b501e`;
  nur die oben eingefrorenen Kandidatenidentitäten und Schema-Namen werden
  durch `2026-08-14-audio-cpp-nonspeech-harness.patch`, SHA-256
  `ca10bc2f056a98a67f48e70f1be3925401bc03875564bc4c92e2f8e2d3f71fc4`,
  ersetzt

Zuerst laufen die festen zehn Non-Speech-Fälle genau einmal bei Concurrency 1.
Harte Grenzen: `10/10` Erfolg, höchstens fünf halluzinierte Wörter, null
fehlgeformte Antworten, null Non-Transcript-Zusätze, Sprachwechsel,
Command-Risk-Fälle oder normalisierte Nichtdeterminismen sowie bestätigtes
Kandidaten-CUDA. Ein Fehler ergibt terminal `ineligible`; danach findet kein
vollständiges A/B statt.

Nur nach PASS lautet die feste A/B-Reihenfolge: Produktion Qualität 70,
Kandidat Qualität 70, Produktion Last 12 bei Concurrency 4, Kandidat Last 12
bei Concurrency 4. Der Kandidat darf sich bei WER, CER, Macro-WER,
Real-path-Macro-WER, leeren/fehlgeformten Antworten, Safety-Zählern, Entity
Recall sowie Zahlen-, Zeit- und Datumsgenauigkeit nicht verschlechtern. Alle
Requests und CUDA-Gates müssen bestehen; p50-Verhältnis `<=1,05`,
p90-Verhältnis `<=1,10`.

Keine selektiven Retries, Outlier-Entfernung oder Parameterkorrektur. Eine
gepaarte Lastwiederholung ist nur bei ausschließlich gefallenem Latenz-Gate und
zeitgleich unabhängig belegter Shared-Host-Störung zulässig.

## Kontrolliertes Fenster und Abschluss

Vor jedem Stop werden Container-ID, Image, State, Health, Restart-Count,
Restart-Policy, Stack-Zuordnung und anwendungsseitig sichtbarer CUDA-Speicher
gesichert. Es werden nur die zur Ressourcenfreigabe zwingend nötigen
GPU-Dienste temporär gestoppt; Produktion und unbeteiligte Dienste werden in
den exakten Vorzustand zurückgebracht. Kandidatencontainer, Netze, Gateway und
Modellkopie werden anschließend entfernt oder als ausdrücklich versiegeltes
owner-only Beobachtungsartefakt bezeichnet. Der terminale Status ist genau
`eligible`, `ineligible` oder `invalid`.
