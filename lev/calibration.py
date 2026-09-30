"""Platt scaling on the top-1 margin: p(correct) = sigmoid(a*margin + b)."""
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Calibration:
    a: float
    b: float

    def p(self, margin: float) -> float:
        """Probability that the top-1 label is correct for a given margin."""
        z = self.a * margin + self.b
        if z >= 0:
            return 1 / (1 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1 + ez)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps({"a": self.a, "b": self.b}),
                       encoding="utf-8")
        tmp.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(a=float(data["a"]), b=float(data["b"]))

    @classmethod
    def fit(
        cls,
        records: list[tuple[float, bool]],
        iterations: int = 500,
        lr: float = 0.1,
    ) -> "Calibration":
        """Batch gradient descent on log-loss over (margin, was_correct)."""
        a, b = 1.0, 0.0
        n = max(len(records), 1)
        for _ in range(iterations):
            da = db = 0.0
            for margin, correct in records:
                p = cls(a, b).p(margin)
                err = p - (1.0 if correct else 0.0)
                da += err * margin
                db += err
            a -= lr * da / n
            b -= lr * db / n
        return cls(a=a, b=b)
