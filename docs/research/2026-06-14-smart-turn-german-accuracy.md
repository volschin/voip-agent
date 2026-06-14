# Smart Turn v3 — can we reach ~98% German? (research)

**Date:** 2026-06-14
**Question:** After the left-pad fix (German 88.5%→95.0%), can other changes push German turn-detection accuracy toward 98%?
**Short answer:** No, not via config/model choice. The model's published ceiling is ~96.6% German, the test set is synthetic and label-noisy, and our real telephony audio is actually ~92–93%. The only path meaningfully higher is fine-tuning on telephony-band / real German audio.

## Authoritative ceiling (pipecat's own benchmarks, identical model+data)

| Model | German acc | FPR (cut-in) | FNR (wait) |
|---|---|---|---|
| smart-turn-v3.2 **cpu int8** | 96.37% | 2.72% | 0.91% |
| smart-turn-v3.2 **gpu fp32** | 96.60% | 2.19% | 1.21% |

GPU fp32 buys only **+0.23%** on German. 98% is above the model ceiling on this dataset. pipecat also crowdsources dataset cleanup (smart-turn-dataset.pipecat.ai), i.e. the labels have known noise — a measurement ceiling below 100%.

## Our experiments (German test split, our `TurnDetector` code path)

Eval over pipecat's `smart-turn-data-v3.1-test` German subset (1322 samples; 600-sample balanced subset for the matrix). Audio paths built with our real `agent.audio` codec.

| Audio path | acc@0.5 | FP (cut-in) | FN (wait) | best thr |
|---|---|---|---|---|
| ideal float (wideband 16k) | 94.8% | 7.5% | 3.1% | 0.70 → 95.5% |
| our int16 PCM path | 94.8% | 7.5% | 3.1% | identical to float |
| **telephony: 8k → aLaw → 16k (production)** | **92.7%** | **14.5%** | 0.9% | 0.75 → 94.0% |

Findings:
1. **int16 quantization is free** — our PCM path scores identically to ideal float. Not a lever.
2. **Telephony band is the real cost:** going through G.711 aLaw 8 kHz (what the agent actually receives) drops accuracy ~2–3% and **doubles the false cut-in rate** (7.5%→14.5% at thr 0.5). This is the dominant real-world degradation, and it is exactly the failure mode we care about (talking over the caller).
3. **The German test set is 100% synthetic TTS** (datasets chirp3/orpheus/rime; `synthetic=True` for every German row). pipecat's 96.4% — and our numbers — are on synthetic German, not real human callers. Real-caller accuracy is unmeasured and could differ.
4. Residual ~1.4% gap between our wideband (~95%) and pipecat's reported 96.4% is a minor decode/normalization nuance, dwarfed by the telephony effect; not worth chasing.

## Levers, ranked

1. **Threshold ≈ 0.70 (cheap, do it).** On the telephony path, thr 0.70–0.75 gives the best accuracy AND lower false cut-ins than 0.5 (14.5%→~10.6%) at a small wait cost (FN ~1–2%). For a phone agent, fewer talk-overs is the right bias. One config change (`TURN_COMPLETE_THRESHOLD`).
2. **GPU fp32 model: skip.** +0.2% German, not worth the 32 MB / no CPU benefit.
3. **Wideband codec (architectural).** If the SIP leg negotiated G.722 (16 kHz) instead of G.711 aLaw (8 kHz), we'd keep the ~95% instead of ~92% — and STT would get native 16 kHz too. **Checked the config: this is not a one-liner.** `asterisk/pjsip.conf` is `disallow=all` / `allow=alaw`, and the agent is aLaw-narrowband end to end: `_create_external_media` requests `format=alaw`, and `agent/ari.py` / `agent/audio.py` decode via `alaw_decode` + `resample_8k_to_16k` (RTP path assumes 8 kHz aLaw). A wideband switch needs: (a) the Fritzbox trunk to actually offer G.722 (many German SIP/POTS lines are aLaw-only — verify first), (b) `allow=g722` in pjsip, (c) ExternalMedia `format=slin16` so Asterisk transcodes to 16 kHz PCM, (d) rework the agent's in/out audio path to slin16 (drop the 8↔16 k resamples; change the TTS-out path from 24k→8k→aLaw to 24k→16k→slin16). Moderate rework, gated on (a). Separate project.
4. **Fine-tuning (the only path toward ~97–98% on OUR audio).** pipecat open-sources the training script + data. Fine-tune Smart Turn on (a) telephony-band-degraded audio and (b) real German call recordings (non-synthetic). Heavy: needs a GPU, labeled German turn data, and an eval harness — but it's the only realistic route above the stock model's ceiling for our narrowband, real-caller setting.
5. **Ensemble / multi-window (Krisp-style): skip.** Marginal gain, added latency + complexity.

## Recommendation

- Ship the left-pad fix (critical regardless).
- Set `TURN_COMPLETE_THRESHOLD=0.70` for the telephony bias (fewer cut-ins).
- Drop the "98%" target: stock-model ceiling is ~96.6% on synthetic German; our telephony reality is ~92–94%.
- If higher accuracy is genuinely required: (a) check whether G.722 wideband is negotiable on these SIP calls, then (b) plan a fine-tuning effort on telephony-band + real German audio. Both are separate projects.
- Before enabling in production, still do a real-call smoke test — every number here is on synthetic German.
