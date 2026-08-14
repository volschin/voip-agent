# audio.cpp: vollständige Kandidatenqualifizierung

**Stichtag:** 2026-08-14 UTC
**ASR-Status:** `ineligible`
**TTS-Status:** `ineligible`
**Gesamturteil:** weiter beobachten, nicht empfehlen oder adoptieren
**Produktion:** Modelle, Images, Compose und Routen unverändert

## Kurzurteil

`audio.cpp` ist ein technisch interessanter, ungewöhnlich schnell wachsender
lokaler C++/ggml-Audioruntime-Kandidat. Das Projekt besitzt bereits echte
externe Nutzer, ARM64/CUDA-Images, OpenAI-nahe HTTP-Endpunkte und breite
Modellabdeckung. Für den `voip-agent` fällt die eingefrorene Kandidatenidentität
dennoch an zwei produktionskritischen Hard Gates:

1. **ASR:** 10/10 Requests und CUDA bestanden, aber 7 halluzinierte Wörter auf
   den zehn festen Non-Speech-Fällen bei einer Grenze von höchstens 5.
2. **TTS:** der synchrone Offline-Pfad hat keinen kooperativen
   Client-Disconnect-/Cancel-Kanal. Ein abgebrochener Barge-in-Turn kann daher
   bis zum Ende weiterrechnen und den exklusiven Modell-Lock halten.

Die Kandidaten sind damit je Claim terminal `ineligible`. Das ist kein
allgemeines Negativurteil über das Projekt. Auf ausdrücklichen Wunsch wurde
das vollständige ASR-A/B anschließend trotzdem als Charakterisierung
ausgeführt. Die vollständige TTS-Runtime-Charakterisierung ist noch
ausstehend; ihr Hard-Gate-Status wird dadurch nicht erneut entschieden.

## Bewertete Identitäten

### Gemeinsame Runtime

- Source: `0xShug0/audio.cpp@04ba4375ed53bbd718bd2697e190007f6a19f426`
- Image: `full-cuda13-20260814-04ba437`
- OCI-Index:
  `sha256:6d34bf5008c840e03fa279d45b9c02c69308ac7430d2be60d517db4b6362ca0e`
- Linux/arm64-Manifest:
  `sha256:6cddc4a8c306fe07e7baa6d4562d7145777e8eb2664c7c6a45091de67f9881ba`
- OCI-Revision-Label: exakt `04ba4375...`
- CUDA 13, Backend `cuda`, Device 0, ein Thread, lokaler/offline Betrieb

Der verwendete tägliche Build ist nicht einfach das Release-0.6-Tag, sondern
ein exakt bezeichneter späterer Commit. Aussagen gelten nur für diese
Deployment-Identität.

### ASR

- Kandidat: `Qwen/Qwen3-ASR-1.7B-hf`, BF16, Revision
  `bcd2b5b7f32b480ab5790554cfa8347f246a14f3`
- Gewicht: 4.076.193.080 Bytes, SHA-256
  `2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1`
- vollständiger lokaler Manifest-Nachweis:
  `9095527e702f3ebe131ece9521822d1a8b0f4ed957173b8780459ee4c46892bb`
- Produktion: `UrocyonF/Qwen3-ASR-1.7B-NVFP4` Revision `61ad4d...`,
  vLLM-Image `sha256:0fadf01c...`

Das Ergebnis darf nicht als reiner Runtime-Vergleich gelesen werden: Runtime,
Modelllayout und Präzision wechselten gemeinsam von vLLM/NVFP4 zu
audio.cpp/BF16.

### TTS

- unverändertes Modell: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, Revision
  `fd4b254389122332181a7c3db7f27e918eec64e3`
- Voice-Preset: `shared-female-de-v1`
- private Referenz-WAV: SHA-256 `707085e4...`; privater Referenztext nur als
  SHA-256 `f46801ea...`
- Produktionsvertrag: ganze 24-kHz-Mono-PCM16-WAV, danach genau eine
  Konvertierung auf 16 kHz; der Stream-Endpunkt bleibt diagnostisch

Private Voice-Daten wurden nicht ausgegeben, geloggt oder an externe Dienste
übertragen.

## Upstream- und Vertragsaudit

Die offiziellen Modell- und Serverdokumente bestätigen native Unterstützung
für `Qwen3-ASR-1.7B-hf`, Qwen3-TTS Base Voice Cloning und CUDA. Der HTTP-Server
ist jedoch nur OpenAI-**nah**: bei `/v1/audio/transcriptions` und
`/v1/audio/speech` ist `model` zwingend. Die realen Consumer senden heute kein
`model`; ASR erwartet außerdem nur `{text}`, während `audio.cpp` ein additives
`timing` liefert.

Für den ASR-Lauf wurde deshalb der bestehende Consumer-Gateway mit genau zwei
eingefrorenen Änderungen verwendet: `model=qwen3-asr` ergänzen und additives
`timing` beim Normalisieren ignorieren. Patch-SHA-256:
`96f3107904e1db2a706b64e78f1c96f65a4d0087edb052dc20e7a19d1599cd9d`.
Damit blieb der externe Consumer-Vertrag unverändert. Für eine spätere
Produktion wäre dieser Adapter oder eine echte Upstream-Kompatibilitätsoption
ein zusätzlicher zu wartender Bestandteil.

Das Kandidatenimage besitzt keine eingebaute API-Authentifizierung. Es wurde
deshalb nur in einem internen Docker-Netz ohne externen Netzweg betrieben; der
Qualifikations-Gateway war ausschließlich an `127.0.0.1` gebunden. Eine
Adoption müsste weiterhin die authentifizierten Traefik-Routen erzwingen.

## ASR-Lauf

### Startup und Provenienz

- 9/9 offizielle Snapshot-Dateien, keine Symlinks oder Pfadfluchten
- Image-Plattform, Manifest, Entry Point und Revision-Label korrekt
- Server `running|healthy`, Modell `qwen3-asr` eager geladen
- externer Vertrag über den eingefrorenen Gateway bestanden
- Startup-Nachweis SHA-256
  `e5f3afb63bdc7692e2d9cb965e8538107493cde3a13f8f699731e2cf16230287`

### Einmaliger Non-Speech-Discriminator

Corpus und Reihenfolge waren unverändert. Der Harness basierte auf dem bereits
verwendeten Zehn-Fall-Lauf; nur Kandidatenidentitäten und Schema-Namen wurden
vor dem Request ersetzt. Overlay-Patch SHA-256:
`ca10bc2f056a98a67f48e70f1be3925401bc03875564bc4c92e2f8e2d3f71fc4`.

| Gate | Ergebnis | Status |
|---|---:|---|
| erfolgreiche Requests | 10/10 | PASS |
| fehlgeschlagene Requests | 0 | PASS |
| fehlgeformte Antworten | 0 | PASS |
| halluzinierte Wörter | 7 bei Grenze `<=5` | **FAIL** |
| Non-Transcript-Zusätze | 0 | PASS |
| Sprachwechsel | 0 | PASS |
| Command-Risk-Fälle | 0 | PASS |
| normalisierte Nichtdeterminismen | 0 | PASS |
| Kandidaten-CUDA | PID-korreliert, 6 aktive Samples | PASS |

Die Wortanzahlen je neutraler Fall-ID waren `1,1,1,0,0,1,1,1,0,1`.
Der transcriptfreie sichere Nachweis hat SHA-256
`dbfdf366097418da71074f9efd7851e99519d47a1379212bdeacf54f48ff8788`,
das geschützte Rohartefakt
`03b7bfdfcdb8c22364c66758bd0f5eda4550c4c969e090c80abef18b2d1d3d1b`.
Es gab keinen Retry und keine Parameterkorrektur.

**ASR-Status: `ineligible`.** Der Non-Speech-Lauf wurde nicht wiederholt. Auf
ausdrücklichen Nutzerwunsch folgte danach eine getrennt gekennzeichnete
Vollcharakterisierung; sie kann das bereits gefallene Gate nicht heilen.

### Vollständiges Qualitäts- und Last-A/B

Beide Seiten liefen auf derselben eingefrorenen Reihenfolge: jeweils 70
Qualitätsfälle bei Concurrency 1 und danach 12 feste Lastfälle bei Concurrency
4. Alle 164 Requests waren erfolgreich; Produktion und Kandidat zeigten
PID-korrelierte CUDA-Aktivität.

| Qualitätsmetrik | Produktion | Kandidat | Einordnung |
|---|---:|---:|---|
| WER | 0,045276 | 0,041339 | Kandidat besser |
| CER | 0,038239 | 0,037466 | Kandidat besser |
| Macro-WER | 0,061800 | 0,056431 | Kandidat besser |
| Real-path Macro-WER | 0,000000 | 0,000000 | gleich |
| Entity Recall | 0,636364 | 0,636364 | gleich |
| Zahlen | 0,166667 | 0,166667 | gleich |
| Zeiten | 0,166667 | 0,166667 | gleich |
| Daten | 0,000000 | 0,000000 | gleich |
| exakte Groß-/Kleinschreibung | 0,750000 | 0,750000 | gleich |
| exakte Interpunktion | 0,766667 | 0,750000 | Kandidat schlechter |
| leere Speech-Ergebnisse | 0 | 0 | gleich |
| fehlgeformte Antworten | 0 | 0 | gleich |
| Non-Speech-Wörter im 70er-Lauf | 5 | 7 | Kandidat schlechter |
| Zusätze / Sprachwechsel / Command-Risk | 0 / 0 / 0 | 0 / 0 / 0 | gleich |

Die Qualitätsverbesserung ist klein, aber konsistent über WER, CER und
Macro-WER. Sie kompensiert weder die Non-Speech-Regression noch die
Performance:

| Performance | Produktion | Kandidat | Verhältnis Kandidat/Produktion |
|---|---:|---:|---:|
| Qualitätslauf p50 | 6.888,19 ms | 20.476,86 ms | 2,973 |
| Qualitätslauf p90 | 11.156,89 ms | 27.765,39 ms | 2,489 |
| Lastlauf p50 | 406,03 ms | 2.409,54 ms | 5,934 |
| Lastlauf p95 | 679,28 ms | 4.152,43 ms | 6,113 |
| Lastdurchsatz | 17,164 req/s | 2,876 req/s | 0,168 |

Die vorab gesetzten Latenzgrenzen p50 `<=1,05` und p90 `<=1,10` werden klar
verfehlt. Der Kandidat erreicht unter Last nur rund 16,8 % des
Produktionsdurchsatzes.

Transcriptfreie Vergleichsevidenz:

- Produktion Qualität SHA-256 `84d379077252733ad20615e3d2c72458adc61507dc799fb80952720136384067`
- Kandidat Qualität SHA-256 `90c860dbc88f15ed232d6debce1f0f86fba3983fde9ac994f4d30145df17ff13`
- Produktion Last SHA-256 `94e748812a5d2a083784c38a2790b0dc2d9ad2512ed7ba82f82247b5bce3badd`
- Kandidat Last SHA-256 `57e1e0939b004384eafc84e62ea441e38c65265b1af406fffd3940839e12dc8b`
- sicheres Aggregat SHA-256 `05099f9794d1e7b250698ebab20bf4b7852b4c0129cb8aef8b7fe8dccb3fafed`

## TTS-Fail-fast

Der zum Image gehörende Quellstand wurde vor einem Modellstart geprüft:

- `app/server/runtime.cpp` SHA-256 `2d32ea9f...`
- `app/server/busy_guard.h` SHA-256 `9c0216f1...`
- `run_model()` hält den `BusyGuard` während `prepare()` und dem synchronen
  `offline->run()`.
- Der Offline-Speech-Pfad erhält weder Connection-Lifecycle noch Stop-Token.
- Upstream beschreibt selbst, dass eine laufende CUDA-Inferenz nicht aus
  Userspace abgebrochen werden kann; `busy_timeout_ms` weist nur wartende
  Folgerequests mit 503 ab.

Das verletzt das harte Projekt-Invariant: Barge-in muss Playback und
Generation abbrechen, veraltete Ausgabe verwerfen und den exklusiven TTS-Lock
zügig freigeben. Die bestehende Produktion besitzt dafür einen expliziten
kooperativen Cancel-Pfad.

**TTS-Status: `ineligible`.** Die statische Ursache steht fest. WAV-Format,
Voice-Stabilität, deutsche Verständlichkeit, Latenz, Last und der praktische
Abbruch-/Sentinel-Verlauf werden auf ausdrücklichen Nutzerwunsch noch separat
charakterisiert und bleiben bis dahin `unproven`.

## Öffentliche Erfahrungen und Projektreife

Das öffentliche Bild ist überwiegend positiv, aber noch dünn und stark vom
Maintainer geprägt:

- TranscrIA hat `audio.cpp` als erstklassige Engine integriert und berichtet
  für Qwen3-ASR-1.7B auf acht schwierigen französischen Meeting-Ausschnitten
  WER 0,42 sowie etwa 10–14 Sekunden pro fünf Minuten Audio. Gleichzeitig wird
  eine Stutter-Schleife im chaotischsten Multi-Speaker-Fenster genannt. Ein
  Sessionfehler wurde nach externem Test sehr schnell behoben.
- Mehrere LocalLLaMA-Nutzer loben deutlich geringere TTS-Latenz und einfachere
  Deployments; ein Nutzer verwendet es bereits für Agenten-ASR und -TTS. In
  derselben Diskussion wurden aber auch ein CUDA-Docker-Fallback auf CPU und
  ungeeignete Qwen3-TTS-Defaults erwähnt, die erst kurzfristig korrigiert
  wurden.
- Hands-on-Berichte zu Irodori-TTS und ein japanisches Laufprotokoll bewerten
  Bedienbarkeit und Ressourcenbedarf positiv. Sie ersetzen keine
  deutschsprachige Telefon-, Voice-Identity- oder Barge-in-Validierung.
- Am Stichtag war das Repository erst seit 2026-06-23 öffentlich, hatte etwa
  1.414 Stars und 172 Forks. Die Aktivität ist hoch; entsprechend ändern sich
  API, Defaults und Modellpfade noch rasch.

Die belastbarste externe Evidenz betrifft französische Meetings und andere
TTS-Modelle/GPU-Klassen. Es gibt weiterhin keinen unabhängigen Nachweis für
deutsches Telefon-Audio, NVIDIA GB10, die private Voice-ID oder den konkreten
Cancel-Vertrag dieses Projekts.

## Zwischenzustand und Evidenz

- Kandidaten- und Gateway-Container: entfernt
- beide Kandidatennetze: entfernt
- Kandidaten- und Gateway-Images sowie die versiegelte Modellkopie: für die
  noch ausstehende TTS-Charakterisierung lokal vorgehalten, aber nicht laufend
- owner-only Evidenz, Hashmanifest und reproduzierbare Overlay-Quellen:
  `/home/volsch/ai-companion/runtime/audio-cpp-qualification-20260814-04ba437`
- Produktions-ASR: gleiche Container-ID und gleiches Image, `running|healthy`,
  Policy `unless-stopped`
- alle Containerzustände, Images, Health-Werte und Stack-Zuordnungen sind nach
  Ausblendung des Docker-Restartzählers bytegleich; kanonischer SHA-256
  `5829fdb1ae1aeeef57df1c8fd042fa2dbe006e9ef7e0b92a76feff178509396b`
- einzige Metadatenabweichung: der manuelle kontrollierte Stop/Start setzte den
  bisherigen ASR-`RestartCount` von 1 auf 0 zurück. Ein künstlicher Crash nur
  zum Wiederherstellen dieses Zählers wurde bewusst nicht erzeugt.
- laufende Kandidatencontainer/-netze: 0; ungesunde Container: 0

## Beobachtungskriterien für einen neuen Lauf

Eine neue Kandidatenidentität lohnt sich erst, wenn mindestens eines der
folgenden Signale vorliegt:

1. Offline-TTS propagiert Client-Disconnect oder Stop-Token bis in die
   Modellsession und besitzt einen reproduzierbaren Cancel-Test.
2. Die OpenAI-Endpunkte können bei genau einem konfigurierten Modell den
   bestehenden Vertrag ohne separaten Body-Rewrite-Adapter bedienen.
3. Qwen3-ASR erhält einen nachvollziehbaren Fix gegen Non-Speech-Ausgaben oder
   Stutter-Loops, ohne externen Postfilter.
4. Es gibt ARM64/GB10-Inferenztests statt ausschließlich Docker-Builds oder
   RTX-5090-Benchmarks.
5. Externe Tests decken deutsches Narrowband/G.722, Namen, Zahlen, Termine und
   Voice Cloning ab.

Ein späterer Commit, ein GGUF/Q8-Modell, geänderte Sampling-Defaults oder ein
zusätzlicher Filter sind jeweils ein neuer Claim und dürfen diesen Status nicht
überschreiben.
