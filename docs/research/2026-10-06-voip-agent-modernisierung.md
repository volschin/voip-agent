# Modernisierung des VoIP-Agenten: Ideen und Prioritäten

Stand: 6. Oktober 2026. Grundlage: Repository auf `b1fbb47` und aktuelle
Primärquellen. Dieses Dokument hält die Überlegungen aus dem Gespräch fest.
Es ist eine Diskussionsgrundlage, kein freigegebener Implementierungsplan.
Es wurden keine Live-Benchmarks oder Laufzeitprüfungen durchgeführt.

## Einschätzung

Ein schlanker Agent-Harness wie Pi ist ein plausibler Kandidat für die
Gesprächs- und Tool-Steuerung. Den größten unmittelbaren Nutzen versprechen
jedoch natürlicheres Gesprächsverhalten, verlässliche Aufgabenabläufe und
messbare Qualität. Ein Harness allein macht den Telefonassistenten noch
nicht schneller oder hilfreicher.

Der aktuelle Agent bringt bereits Smart Turn v3, Barge-in mit Cancellation,
satzweise TTS und einen begrenzten Tool-Loop mit. Darauf lässt sich aufbauen.
Die Prioritäten unten sind eine Einschätzung für dieses Projekt und keine
auf GX10 nachgewiesenen Leistungsversprechen.

## Pi als Agent-Harness

Pi bietet neben dem Coding-Agent einen separaten Agent-Core mit
Zustandsverwaltung, Tool-Ausführung, Ereignisstream, Kontextbearbeitung und
Abbruchsteuerung. Für den VoIP-Agent wäre dieser Core der relevante Baustein.
Das Projekt liegt inzwischen unter `earendil-works/pi`; frühere
`badlogic/pi-mono`-Links werden dorthin umgeleitet.

Möglicher Nutzen:

- Den selbst geschriebenen LLM-/Tool-Loop vereinheitlichen.
- Fähigkeiten wie Kalender, Wissenssuche und Nachrichtenaufnahme gezielt anbieten.
- Tool-Fortschritt in passende gesprochene Rückmeldungen übersetzen.
- Gesprächskontext kompakt halten und weitere Kanäle anbinden.

Der Preis wäre eine zusätzliche TypeScript-/Node-Laufzeit neben Python.
Pi-Steering ist außerdem keine fertige Barge-in-Lösung: Laut Dokumentation
wird Steering nach Abschluss der laufenden Tool-Runde verarbeitet.
Sofortiger Audioabbruch, veraltete Antworten und bereits gestartete
Schreiboperationen müssten weiterhin ausdrücklich behandelt werden.

Die bestehende Session-FSM, Mediensteuerung, serverseitige
Kalenderbestätigung und Anruferberechtigung bleiben maßgeblich. Ein
Harness-Abbruch macht eine bereits ausgeführte externe Aktion nicht rückgängig.

Quellen: [Pi](https://github.com/earendil-works/pi),
[Agent-Core-Dokumentation](https://github.com/earendil-works/pi/blob/main/packages/agent/README.md).

## Interessante Entwicklungen und ihr möglicher Mehrwert

| Ansatz | Konkreter Mehrwert | Priorität |
|---|---|---|
| Semantische Unterbrechungen | „Mhm“ oder Husten beendet nicht gleich die Antwort; „Stopp, anderer Termin“ unterbricht zuverlässig. | Sehr hoch |
| Aufgabenorientierte Gesprächsabläufe | Terminvereinbarung, Rückrufwunsch und Nachrichtenaufnahme ergänzen fehlende Angaben und kommen zu einem klaren Abschluss. | Sehr hoch |
| Gesprächssimulationen und Outcome-Evals | Prüfen, ob der richtige Termin tatsächlich angelegt wurde und eine Ablehnung jede Schreibaktion verhindert. | Sehr hoch |
| Latenzmessung pro Gesprächsschritt | Zeigt, ob Wartezeit durch Turn-Erkennung, ASR, LLM, Tools oder erste TTS-Ausgabe entsteht. | Sehr hoch |
| Streaming-ASR und spekulative Vorbereitung | Während der Anrufer spricht, bereits vorläufig transkribieren und eine Antwort vorbereiten. | Hoch, nach Messung |
| Besseres Retrieval | Namen, Abkürzungen und konkrete Fakten zuverlässiger finden; bei schwachen Treffern nachfragen. | Hoch |
| Selektive Fähigkeiten und Tools | Pro Aufgabe und Berechtigung nur relevante Tools laden. | Hoch |
| Strukturierte Gesprächsergebnisse | Anliegen, Rückrufnummer, Terminwunsch und offene Punkte als nutzbare Daten erfassen. | Hoch |
| Gedächtnis über mehrere Anrufe | Bestätigte Präferenzen und offene Anliegen wiedererkennen. | Mittel |
| Hintergrundaufgaben nach dem Anruf | Zusammenfassung oder längere Recherche fortsetzen, ohne den Anrufer warten zu lassen. | Mittel |
| WebRTC-Testoberfläche | Vom iPad aus sprechen, Unterbrechungen testen und die Verarbeitung beobachten. | Mittel |
| Native Speech-to-Speech / Full Duplex | Perspektivisch flüssigere Übergänge, gleichzeitiges Zuhören und Sprechen, natürlichere Stimme. | Forschungszweig |

### 1. Natürlicheres Gesprächsverhalten

Aktuelle Voice-Systeme unterscheiden zunehmend zwischen echter
Unterbrechung, kurzen Zustimmungen und Hintergrundsprache. NVIDIA beschreibt
Backchannel-Unterdrückung; LiveKit unterstützt die Wiederaufnahme nach einer
falschen Unterbrechung.

Für den Agenten wären zusätzlich sinnvoll:

- Eine Rückfrage pro Antwort und kurze, gut sprechbare Sätze.
- Sauber ausgesprochene Uhrzeiten, Datumsangaben und Telefonnummern.
- Konkrete Fortschrittsmeldungen wie „Ich prüfe die freien Termine“.
- Sorgfältige Unterscheidung zwischen „Mhm“, Hintergrundgeräusch und „Stopp“.

Quellen: [NVIDIA Turn-Taking und Backchannels](https://docs.nvidia.com/nemo/labs-voice-agent/about/core-concepts/speech-pipeline/turn-taking-backchannels/),
[LiveKit Turns und Unterbrechungen](https://docs.livekit.io/agents/logic/turns/).

### 2. Agentische Flexibilität mit festen Aufgabenabläufen kombinieren

Der Telefonassistent profitiert von Flexibilität bei Sprache und Anliegen,
aber auch von einem klaren Ablauf:

```text
Anliegen verstehen → Angaben ergänzen → Vorschlag vorlesen
                   → Bestätigung erhalten → Aktion ausführen
```

Das lässt sich mit einem kleinen Harness umsetzen. Ein eigenes
Expertenagenten-Team für Kalender, RAG und Gesprächsführung hat zunächst
niedrige Priorität: Der zusätzliche Koordinationsaufwand muss einen
konkreten Nutzen zeigen.

### 3. Streaming-ASR und spekulative Verarbeitung

LiveKit unterstützt vorgezogene LLM-Generierung vor dem bestätigten
Turn-Ende. Dabei entstehen zusätzliche, gegebenenfalls verworfene
Modellaufrufe.

Für den Agenten wäre das ein späterer Versuch: vorläufige Transkription und
lesende Vorbereitung erlauben, bei einer Fortsetzung sofort verwerfen.
Schreibende Tools dürfen daraus noch keine Aktion ableiten. Zuerst muss
gemessen werden, welcher Teil der heutigen Pipeline tatsächlich dominiert.
Die Eignung des vorhandenen ASR-Vertrags für Streaming ist noch zu prüfen.

Quelle: [LiveKit AgentSession](https://docs.livekit.io/agents/logic/sessions/).

### 4. Retrieval verbessern

Aktuell liefert `agent/tools/rag.py` die fünf nächsten Vektortreffer als
Text zurück. Mögliche gezielte Verbesserungen:

- Volltext- und Vektorsuche kombinieren.
- Quelle, Aktualität und Relevanz mitliefern.
- Schwache Treffer erkennen und nachfragen.
- Bei Bedarf die Treffer neu bewerten.

Gerade bei deutschen Eigennamen, Produktbezeichnungen und Abkürzungen könnte
das mehr bringen als ein größeres Gesprächsmodell. Es wäre mit dem vorhandenen
PostgreSQL/pgvector-Aufbau untersuchbar; ein Qualitätsgewinn ist noch nicht
gemessen.

### 5. Ganze Gespräche und ihre Ergebnisse evaluieren

Ein aussagekräftiger Testfall wäre: „Dienstag um zehn – nein, doch Mittwoch“,
danach eine Unterbrechung und schließlich eine Bestätigung. Geprüft werden
Gesprächsverhalten und der resultierende Kalenderzustand.

Weitere Fälle: Ablehnung eines Vorschlags, Auflegen während einer
Tool-Ausführung, unbekannter Anrufer, längere Denkpause und Hintergrundsprache.
Deterministische Zustandsprüfungen ergänzen sprachliche Bewertungen.
LiveKit bietet mehrstufige Agent-Simulationen mit Ergebnisprüfungen.

Die vorhandene Telemetrie in `agent/observability.py` erfasst bereits
Stufenzeiten und Fehler. Ergänzen ließen sich insbesondere:

- Zeit vom bestätigten Turn-Ende bis zum ersten hörbaren Audio.
- Latenzverteilungen einschließlich p95 statt nur Summe und Anzahl.
- Reaktionszeit und Fehlalarme bei Unterbrechungen.
- Erfolgreiche Aufgabenabschlüsse und korrekt abgebrochene Aktionen.

Mock-basierte Repository-Tests bleiben von gesonderten Modell-Evals und
Live-Telefonieprüfungen getrennt.

Quelle: [LiveKit Agent Simulations](https://docs.livekit.io/testing/simulations/).

### 6. Strukturierte Ergebnisse, Gedächtnis und Hintergrundaufgaben

Ein strukturierter Gesprächsabschluss könnte Anliegen, bestätigte Angaben,
Ergebnis und offene Punkte erfassen. Daraus lassen sich später gezielte
Benachrichtigungen oder Nachbearbeitung ableiten.

Gedächtnis über mehrere Anrufe wäre vor allem für bestätigte Präferenzen und
offene Anliegen interessant. Dabei müssen berechtigte Identität, Herkunft,
Korrektur und Löschung der Einträge festgelegt werden; beliebige Aussagen
sollten nicht automatisch zu dauerhaftem Wissen werden.

Längere Recherche oder Zusammenfassungen könnten nach dem Anruf als
Hintergrundaufgaben weiterlaufen. Falls solche Aufgaben tatsächlich benötigt
werden, ist dauerhafte Ausführung ein relevanter Baustein. Pydantic AI
dokumentiert entsprechende Integrationen. Für den zeitkritischen Audiopfad
ist daraus noch keine Empfehlung für einen Workflow-Motor abgeleitet.

Quelle: [Pydantic AI Durable Execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/).

## Architekturvarianten zum Vergleich

| Variante | Warum sie interessant ist | Aufwand |
|---|---|---|
| Bestehendes Python-System mit kleinem eigenen Harness weiterentwickeln | Nutzt vorhandene Mediensteuerung und Verträge; Änderungen bleiben gezielt. | Geringster |
| Pydantic AI für LLM und Tools | Python, typisierte Tools und Ausgaben, Streaming und Testunterstützung. Naheliegender Vergleichskandidat für diesen Stack. | Mittel |
| Pi Agent-Core für LLM und Tools | Interessant als gemeinsame Agent-Laufzeit für Telefon, Chat und weitere Kanäle. | Mittel bis höher |
| Pipecat oder LiveKit als umfassende Voice-Basis | Sinnvoll bei mehreren Audiokanälen oder wenn mehr Voice-Verhalten vom Framework übernommen werden soll. | Höher |

Die Aufwandseinschätzung bezieht sich auf dieses Repository und ist keine
allgemeine Produkteigenschaft. Pipecats lokal ausgeführter Smart Turn v3
steckt bereits im Agenten; ein Frameworkwechsel wäre dafür nicht erforderlich.
Ein weitergehender Wechsel müsste insbesondere die direkte FRITZ!Box-
PJSUA2-Anbindung und die bestehenden Medien- und Cancellation-Verträge prüfen.

Quellen: [Pydantic AI](https://pydantic.dev/docs/ai/overview/),
[Pipecat](https://docs.pipecat.ai/pipecat/learn/overview),
[LiveKit Agents](https://docs.livekit.io/agents/).

## Full Duplex als separater Forschungszweig

PersonaPlex ist ein Beispiel für native Speech-to-Speech mit Rollen- und
Stimmsteuerung. Die offizielle Modellkarte beschreibt Englisch; daraus
ergibt sich noch keine Eignung für einen deutschen Telefonassistenten oder
für die konkrete GX10-Laufzeit.

Vor einer Übernahme wären deutsche Verständlichkeit, Telefonie-Audio,
Tool-Nutzung, Bestätigungen, Unterbrechungen und tatsächliche Hardware-
Kompatibilität isoliert zu qualifizieren. Modellwechsel und Runtime-Wechsel
müssen bei Vergleichen klar zugeordnet werden.

Quellen: [PersonaPlex](https://github.com/NVIDIA/personaplex),
[Modellkarte](https://huggingface.co/nvidia/personaplex-7b-v1).

## Empfohlene Reihenfolge

1. Gesprächsqualität und Evals verbessern; heutige Latenzen und Fehler messen.
2. Zwei konkrete Aufgabenabläufe ausbauen, beispielsweise Terminvereinbarung
   und Nachrichtenaufnahme.
3. Den heutigen Tool-Loop gegen Pi und Pydantic AI auf denselben Szenarien
   vergleichen: Ergebnisqualität, Cancellation, Latenz und Wartungsaufwand.
4. Streaming-ASR und spekulative Verarbeitung anhand der gemessenen
   Engpässe untersuchen.
5. Full Duplex getrennt als Forschungszweig verfolgen.

Die bisher vereinbarte Trennung von Telefonie und GX10-Diensten bleibt der
Ausgangspunkt. Es ist noch keine Architektur ausgewählt oder Umsetzung
freigegeben. Ein neuer Harness sollte einen messbaren Vorteil gegenüber dem
vorhandenen System zeigen.
