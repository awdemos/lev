import pytest

from lev.loop import classify
from lev.types import ChoiceQuestion

QUESTION = ChoiceQuestion(
    question="Which department should handle this?",
    labels={
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "other": "everything else",
    },
)

onnxruntime = pytest.importorskip("onnxruntime")
huggingface_hub = pytest.importorskip("huggingface_hub")

from lev import onnx_backend  # noqa: E402


def test_signal_name():
    assert onnx_backend.OnnxEmbeddingSignal().name == "onnx_embedding"


def test_cosine_ordering_billing_above_technical():
    sig = onnx_backend.OnnxEmbeddingSignal()
    scores = sig.score("please refund my invoice", {}, QUESTION)
    assert scores["billing"] > scores["technical"]
    assert scores["billing"] > scores["other"]


def test_embeddings_normalized():
    model = onnx_backend._load_model()
    emb = model.encode(["hello world"])[0]
    assert abs(sum(x * x for x in emb) - 1.0) < 1e-4
