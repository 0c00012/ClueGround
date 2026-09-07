from __future__ import annotations

import os
from pathlib import Path


TEXT_MODEL_REVISIONS = {
    "bert-base-uncased": "86b5e0934494bd15c9632b12f734a8a67f723594",
    "medicalai/ClinicalBERT": "f7c7f65227cb311f33a79c24858d875876d478ac",
    "roberta-base": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
    "thomas-sounack/BioClinical-ModernBERT-base": (
        "c3648aa87af95837c809e6f0c5f85d08160db437"
    ),
}


def enable_offline_hf() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def pinned_snapshot(repo_id: str, *, require_main_ref: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    revision = TEXT_MODEL_REVISIONS[repo_id]
    snapshot = Path(
        snapshot_download(repo_id, revision=revision, local_files_only=True)
    ).resolve()
    if snapshot.name != revision:
        raise RuntimeError(
            f"Unexpected Hugging Face snapshot for {repo_id}: {snapshot.name} != {revision}"
        )
    if require_main_ref:
        main_snapshot = Path(
            snapshot_download(repo_id, revision="main", local_files_only=True)
        ).resolve()
        if main_snapshot.name != revision:
            raise RuntimeError(
                f"Cached main ref for {repo_id} is not pinned: {main_snapshot.name} != {revision}"
            )
    return snapshot


def pin_record(repo_id: str, *, require_main_ref: bool = False) -> dict[str, str]:
    snapshot = pinned_snapshot(repo_id, require_main_ref=require_main_ref)
    return {
        "repo_id": repo_id,
        "revision": TEXT_MODEL_REVISIONS[repo_id],
        "snapshot_path": str(snapshot),
    }


def install_legacy_bert_safetensors_loader(snapshot: str | Path) -> None:
    """Let pytorch-pretrained-bert read a pinned safetensors-only snapshot."""
    import torch
    from safetensors.torch import load_file

    snapshot_path = Path(snapshot).resolve()
    safetensors_path = snapshot_path / "model.safetensors"
    pytorch_path = snapshot_path / "pytorch_model.bin"
    if pytorch_path.is_file() or not safetensors_path.is_file():
        return

    previous_load = torch.load

    def compatible_load(source, *args, **kwargs):
        try:
            source_path = Path(source).resolve()
        except (TypeError, OSError):
            source_path = None
        if source_path == pytorch_path:
            return load_file(str(safetensors_path), device="cpu")
        return previous_load(source, *args, **kwargs)

    torch.load = compatible_load
