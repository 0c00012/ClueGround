from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def bert_hidden_size(model: Any) -> int:
    """Read BERT width from the loaded model instead of its path-like name."""
    value = getattr(getattr(model, "config", None), "hidden_size", None)
    if value is None:
        embeddings = getattr(getattr(model, "embeddings", None), "word_embeddings", None)
        weight = getattr(embeddings, "weight", None)
        shape = getattr(weight, "shape", ())
        value = shape[-1] if len(shape) >= 2 else None
    try:
        hidden_size = int(value)
    except (TypeError, ValueError):
        hidden_size = 0
    if hidden_size <= 0:
        raise RuntimeError("Could not determine the loaded TransVG BERT hidden size")
    return hidden_size


def install_transvg_bert_hidden_size_compat(audit_path: str | Path) -> None:
    """Fix TransVG's model-name heuristic for pinned local BERT snapshots."""
    from models.language_model import bert as transvg_bert

    bert_class = transvg_bert.BERT
    if getattr(bert_class, "_pinned_hidden_size_compat", False):
        return
    original_init = bert_class.__init__
    destination = Path(audit_path).resolve()

    def compatible_init(self, name, train_bert, hidden_dim, max_len, enc_num):
        original_init(self, name, train_bert, hidden_dim, max_len, enc_num)
        legacy_declared = int(self.num_channels)
        detected = bert_hidden_size(self.bert)
        self.num_channels = detected
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "bert_source": str(name),
                    "legacy_name_heuristic_width": legacy_declared,
                    "loaded_model_hidden_size": detected,
                    "corrected": legacy_declared != detected,
                    "reason": (
                        "TransVG treats every name other than the literal bert-base-uncased "
                        "as 1024-wide; pinned local bert-base snapshots are 768-wide."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    bert_class.__init__ = compatible_init
    bert_class._pinned_hidden_size_compat = True

