# vLLM 0.26.1rc1 ohne internen Language-Hint

**Datum:** 2026-08-10 UTC
**Entscheidung:** Nicht rolloutfähig

## Fragestellung

Der frühere Vergleich zwischen Produktion vLLM 0.23 und dem Kandidaten
vLLM 0.26.1rc1 zeigte bei identischem Qwen3-ASR-1.7B-Modell eine
Non-Speech-Regression von `5` auf `14` halluzinierte Wörter. Dieser Folgetest
prüft, ob die Regression durch das an vLLM weitergereichte Multipart-Feld
`language=de` verursacht wird und ob der Kandidat ohne diesen internen Hint
weiterhin die Qualitäts- und Performance-Gates besteht.

## Aufbau und Provenienz

Getestet wurde exakt der frühere Kandidat:

- Produktion: `sha256:0fadf01c8957a91ad83aca03395e7cd61fb66c1b20f5049e268ddd5424560930`
- Kandidat: `sha256:ccbee8c22f1619e35ff5f56244d371c663d22628ee919e016a3fec9b535b0fb0`
- Gateway: `sha256:f434a9c6533dd050968e56f547206ca4660ab4aa53d2f9d85ee6b5d8f12f473c`
- Modellrevision: `61ad4d533c64e033a750b66c44aad6f18634997e`
- Corpus: `asr-companion-de-v1+4519e93d3f8f6b99`

Das ursprüngliche Kandidaten-Tag war nicht mehr vorhanden. Der Kandidat wurde
ohne Pull aus dem weiterhin lokal vorhandenen, fest gepinnten Basis-Digest und
der historischen Dockerfile rekonstruiert. Der Docker-Build-Cache erzeugte
wieder exakt dieselbe Image-ID `ccbee8c...`. Runtime-Versionen und ASR-Adapter
wurden vor dem Test verifiziert.

Die Benchmark-Clients sendeten weiterhin `language=de` an den unveränderten
Gateway-Vertrag. Nur im Kandidaten-Gateway wurde mit einem read-only
eingebundenen Einzeiler-Overlay das interne Multipart-Feld `language` auf dem
Weg zum vLLM-Backend unterdrückt. Modell, Audiodaten, Reihenfolge, Concurrency,
Laufzeitparameter, Gateway-Image und Antwortnormalisierung blieben
unverändert.

Relevante SHA-256-Identitäten:

- historische Dockerfile: `c1f5282ee9f6820f6085e9c4ee56616793cf195b2cbadc3f93766686aa5f3130`
- Benchmark-Runner: `091c57013d1422c09bacf72060b95a593a91a8171661d4f6ebe80d29c218cdb1`
- Full-A/B-Treiber: `dd3676563fb156f6075e25379df1a5af4e22df78575499a6ebd466dafcd0558d`
- Comparator: `4887d6a5e5958c47e2bf865bd81fe892ec8293708f888e04b6793373303d2deb`
- Gateway-Client Produktion: `6bcfec719b49b5cefe540193ced78ecf45b6a77c0e572a2a70cf7eccbbfd6cc7`
- Gateway-Client ohne Hint: `52faa21b23e27eb87b241b80a5f0403ae8b590fc2bc5997b0b55a69714e51c69`
- geschütztes Manifest: `44a3d411e0adcd15a740a9f59ee6aaf52d2946fc3956be08908ab9c516982663`
- Last-Subset: `356485dd63ed79fcb4d0bbb8d17063fa8361c3a072c1cfd6f0eae0b82eae859d`
- sicherer Non-Speech-Nachweis: `ff07c535a86774bd89a6d29b5f6f21f8bd9144f6880df8c6169cf8575900b8ba`
- sicherer Full-A/B-Nachweis: `f03213c70edc938e4a2dfc405fa4779efae2fa0fbf8873ec053804f412ccf949`

## Vorgeschalteter Non-Speech-Test

Der feste Zehn-Fall-Test wurde genau einmal mit Concurrency `1` ausgeführt:

- Anfragen: `10/10`, Fehler `0`
- Non-Speech-Wörter: `5`; Gate `<=5` bestanden
- Verteilung: `1,1,1,0,0,0,0,1,0,1`
- fehlgeformte Antworten: `0`
- Non-Transcript-Zusätze: `0`
- Sprachwechsel: `0`
- Command-Risiko: `0`
- normalisierte Nichtdeterminismen: `0`
- Kandidaten-CUDA: PID-korreliert und aktiv

Damit war die Bedingung für das vollständige Testprogramm erfüllt.

## Vollständiges A/B

Alle Serien liefen genau einmal in der festgelegten Reihenfolge:

- Produktion Qualität: `70/70`
- Kandidat Qualität: `70/70`
- Produktion Last: `12/12`, Concurrency `4`
- Kandidat Last: `12/12`, Concurrency `4`
- fehlgeschlagene Anfragen: `0`
- CUDA: beidseitig PID-korreliert, maximal jeweils `96%` SM

### Qualität

| Metrik | Produktion v0.23 | v0.26.1rc1 ohne Hint | Gate |
|---|---:|---:|---|
| WER | `0.04330708661417323` | `0.045275590551181105` | FAIL |
| CER | `0.03785245268443414` | `0.038238702201622246` | FAIL |
| Macro WER | `0.05783146591970121` | `0.061799719887955185` | FAIL |
| Real-path Macro WER | `0.0` | `0.0` | PASS |
| Entity Recall | `0.6363636363636364` | `0.6363636363636364` | PASS |
| Non-Speech-Wörter | `5` | `5` | PASS |

Zahlen-, Zeit- und Datumsgenauigkeit, Groß-/Kleinschreibung, Interpunktion und
alle Safety-Zähler waren identisch. Das Qualitäts-Gate fällt dennoch wegen
WER, CER und Macro WER.

### Performance

| Metrik | Produktion v0.23 | v0.26.1rc1 ohne Hint | Verhältnis | Grenze | Gate |
|---|---:|---:|---:|---:|---|
| p50 | `390.745 ms` | `391.004 ms` | `1.000663` | `<=1.05` | PASS |
| p90 | `552.231 ms` | `554.395 ms` | `1.003917` | `<=1.10` | PASS |
| Durchsatz | `18.124 req/s` | `18.453 req/s` | – | – | INFO |

Die Performance ist praktisch identisch und besteht beide Latenz-Gates.

## Einordnung

Das Unterdrücken des internen `language=de` beseitigt bei v0.26.1rc1 die
Non-Speech-Regression von historisch `14` auf `5` Wörter, tauscht sie aber
gegen eine kleine Verschlechterung der deutschen Transkriptionsqualität ein.
Die Qualitätswerte entsprechen exakt den bereits beobachteten v0.24-Werten
ohne internen Language-Hint. Das spricht dafür, dass der Hint beide
vLLM-Versionen in dieselben zwei Betriebszustände verschiebt:

- mit Hint: bessere deutsche Transkriptionsmetriken, aber Non-Speech-Regression;
- ohne Hint: sauberes Non-Speech-Verhalten, aber leicht schlechtere WER/CER.

Der Kandidat ist wegen des Qualitäts-Gates **nicht rolloutfähig**. Eine
Wiederholung war nicht zulässig, weil die Replikationsregel nur für einen
isolierten Shared-Host-Latenzausreißer nach bestandener Qualität gilt.

## Abschlusszustand

Alle Testcontainer wurden entfernt. Die Rohartefakte wurden wiederherstellbar
in den Papierkorb verschoben; dieses Dokument enthält keine Transkripte,
Audiodaten oder geschützten Pfade. Das rekonstruierte Kandidaten-Image wurde
behalten.

Produktion wurde nicht gestoppt, neu erstellt oder verändert. Der abschließende
Zustand war `running|healthy`, Neustarts `0`, Image
`sha256:0fadf01c8957a91ad83aca03395e7cd61fb66c1b20f5049e268ddd5424560930`
und Restart-Policy `unless-stopped`.
