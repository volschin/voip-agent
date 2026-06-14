import asyncio

import numpy as np

_SAMPLE_RATE = 16000
_MAX_SAMPLES = 8 * _SAMPLE_RATE  # model accepts up to 8 s


class TurnDetector:
    """In-process Smart Turn v3 end-of-turn classifier.

    Downloads a revision-pinned ONNX model once at construction and runs it on
    CPU (or a configured execution provider). `classify` extracts Whisper
    log-mel features and runs inference off the event loop. Same async
    `classify(pcm) -> bool` interface as the prior HTTP client, so the ARI
    gating path is unchanged.

    Tests inject `session` + `feature_extractor` to avoid a model download.
    """

    def __init__(
        self,
        model_repo: str,
        model_filename: str,
        model_revision: str,
        providers: list[str],
        threshold: float = 0.5,
        session=None,
        feature_extractor=None,
    ) -> None:
        self._threshold = threshold
        if session is not None and feature_extractor is not None:
            self._session = session
            self._fx = feature_extractor
            return
        # Heavy deps imported lazily so importing this module (and running the
        # mocked tests) doesn't require onnxruntime/transformers, and so the
        # download cost is paid only when the feature is actually enabled.
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor

        path = hf_hub_download(
            repo_id=model_repo, filename=model_filename, revision=model_revision
        )
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1  # never starve the RTP pacing loop
        self._session = ort.InferenceSession(path, sess_options=so, providers=providers)
        self._fx = WhisperFeatureExtractor(chunk_length=8)

    def aclose(self) -> None:
        # No external resources to release; kept for call-site symmetry.
        pass

    async def classify(self, pcm_16k: np.ndarray) -> bool:
        """True = caller's turn is complete. Raises on inference failure."""
        return await asyncio.to_thread(self._classify_sync, pcm_16k)

    def _classify_sync(self, pcm_16k: np.ndarray) -> bool:
        audio = pcm_16k[-_MAX_SAMPLES:].astype(np.float32) / 32768.0
        inputs = self._fx(
            audio,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=_MAX_SAMPLES,
            truncation=True,
            do_normalize=True,
        )
        outputs = self._session.run(None, {"input_features": inputs.input_features})
        prob = float(np.asarray(outputs[0]).reshape(-1)[0])
        return prob > self._threshold
