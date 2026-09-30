"""Optional ONNX sentence-embedding signal (install lev[onnx]).

Embeds the state and each label's `label + criteria` text with
onnx-community/all-MiniLM-L6-v2-ONNX (~22M params, downloaded once via
huggingface_hub) and scores by cosine similarity.
"""
from functools import lru_cache
from pathlib import Path

from lev.types import ChoiceQuestion

MODEL_REPO = "onnx-community/all-MiniLM-L6-v2-ONNX"
MODEL_PATH = "onnx/model.onnx"

try:
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
except ImportError:  # pragma: no cover - default env has no extras
    np = ort = snapshot_download = Tokenizer = None


class _OnnxModel:
    """Tokenizer + ONNX session with mean-pooled, L2-normalized embeddings."""

    def __init__(self, model_dir: str) -> None:
        self._tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            f"{model_dir}/{MODEL_PATH}", opts, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        enc = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        types = np.zeros_like(ids)
        out, = self._session.run(
            None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
        )
        # Mean pool over the token axis (standard MiniLM/sentence-transformers
        # usage), masking padding, then L2-normalize so cosine == dot product.
        m = mask.astype(out.dtype)[..., None]
        pooled = (out * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
        norms = np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        return (pooled / norms).tolist()


@lru_cache(maxsize=1)
def _load_model() -> _OnnxModel:
    if ort is None:
        raise RuntimeError(
            "install lev[onnx] (uv pip install 'lev[onnx]') to use the onnx backend"
        )
    try:
        # local_dir materializes real files (HF cache symlinks break ORT's
        # external-data resolution for models split into *.onnx_data).
        model_dir = snapshot_download(
            MODEL_REPO,
            allow_patterns=["onnx/model.onnx", "onnx/model.onnx_data",
                            "tokenizer.json", "config.json"],
            local_dir=str(Path.home() / ".cache" / "lev" / MODEL_REPO.split("/")[-1]),
        )
    except Exception as exc:
        raise RuntimeError(
            f"could not download {MODEL_REPO} (offline?): {exc}"
        ) from exc
    return _OnnxModel(model_dir)


class OnnxEmbeddingSignal:
    """Cosine similarity between state and label embeddings, scaled x5 so raw
    magnitudes stay comparable to the heuristic signals' (EVIDENCE_WEIGHT is
    3.0; cosine in [0, 1] would otherwise be drowned after the softmax)."""

    name = "onnx_embedding"

    def __init__(self) -> None:
        self._model = _load_model()

    def score(
        self, text: str, tokens, question: ChoiceQuestion
    ) -> dict[str, float]:
        texts = [f"{label} {criteria}" for label, criteria in question.labels.items()]
        state, *targets = self._model.encode([text, *texts])
        return {
            label: 5.0 * sum(s * t for s, t in zip(state, target))
            for (label, _), target in zip(question.labels.items(), targets)
        }
