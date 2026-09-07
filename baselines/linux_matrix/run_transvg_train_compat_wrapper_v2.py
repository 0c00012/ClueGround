from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_repro.hf_pins import (  # noqa: E402
    enable_offline_hf,
    install_legacy_bert_safetensors_loader,
    pin_record,
)
from src.baseline_repro.transvg_compat import (  # noqa: E402
    install_transvg_bert_hidden_size_compat,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--lr-drop", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-query-len", type=int, default=20)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_log = output / "checkpoint_save_events.jsonl"
    latest_val_stats: dict[str, float] = {}
    best_miou = float("-inf")
    best_miou_epoch: int | None = None
    enable_offline_hf()
    text_pin = pin_record("bert-base-uncased", require_main_ref=True)
    (output / "hf_text_encoder_pin.json").write_text(
        json.dumps(text_pin, indent=2), encoding="utf-8"
    )
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    import torch
    import utils.misc as misc_utils

    old_load = torch.load
    torch.load = lambda *a, **kw: old_load(*a, **({"weights_only": False} | kw))
    install_legacy_bert_safetensors_loader(text_pin["snapshot_path"])
    install_transvg_bert_hidden_size_compat(output / "bert_hidden_size_compat.json")

    def append_event(row: dict) -> None:
        with save_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}) + "\n")

    def atomic_save(payload: object, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.stem}.tmp_{os.getpid()}{destination.suffix}")
        torch.save(payload, temporary, _use_new_zipfile_serialization=False)
        torch.load(temporary, map_location="cpu")
        os.replace(temporary, destination)

    def safe_save_on_master(payload: object, path_like: object, *unused_args, **unused_kwargs) -> None:
        nonlocal best_miou, best_miou_epoch
        if not misc_utils.is_main_process():
            return
        requested = Path(path_like).resolve()
        if requested.name.startswith("checkpoint0"):
            append_event({"requested": str(requested), "status": "skipped_periodic_checkpoint"})
            return
        destination = requested.with_name("last.pth" if requested.name == "checkpoint.pth" else requested.name)
        if isinstance(payload, dict) and "model" in payload:
            payload = {
                "model": payload["model"],
                "epoch": payload.get("epoch"),
                "args": payload.get("args"),
                "val_accu": payload.get("val_accu"),
                "val_miou": latest_val_stats.get("miou"),
                "note": "Model-only atomic checkpoint written by run_transvg_train_compat_wrapper_v2.py",
            }
        try:
            atomic_save(payload, destination)
            append_event(
                {
                    "requested": str(requested),
                    "saved": str(destination),
                    "status": "ok",
                    "bytes": destination.stat().st_size,
                }
            )
            epoch = payload.get("epoch") if isinstance(payload, dict) else None
            val_miou = latest_val_stats.get("miou")
            if (
                destination.name == "last.pth"
                and epoch is not None
                and val_miou is not None
                and int(epoch) != best_miou_epoch
                and float(val_miou) > best_miou
            ):
                best_miou = float(val_miou)
                best_miou_epoch = int(epoch)
                miou_destination = destination.with_name("best_miou_checkpoint.pth")
                atomic_save(payload, miou_destination)
                append_event(
                    {
                        "requested": str(requested),
                        "saved": str(miou_destination),
                        "status": "best_val_miou_ok",
                        "epoch": best_miou_epoch,
                        "val_miou": best_miou,
                        "bytes": miou_destination.stat().st_size,
                    }
                )
        except Exception as exc:
            append_event({"requested": str(requested), "status": "failed", "error": repr(exc)})
            raise

    misc_utils.save_on_master = safe_save_on_master

    import train as transvg_train

    original_validate = transvg_train.validate

    def tracked_validate(*validate_args, **validate_kwargs):
        stats = original_validate(*validate_args, **validate_kwargs)
        latest_val_stats.clear()
        latest_val_stats.update(
            {
                key: float(value.item()) if hasattr(value, "item") else float(value)
                for key, value in stats.items()
            }
        )
        return stats

    transvg_train.validate = tracked_validate

    train_args = transvg_train.get_args_parser().parse_args(
        [
            "--model_name",
            "TransVG",
            "--seed",
            str(args.seed),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            "0.0001",
            "--lr_bert",
            "0.00001",
            "--lr_visu_cnn",
            "0.00001",
            "--lr_visu_tra",
            "0.00001",
            "--aug_crop",
            "--aug_scale",
            "--aug_translate",
            "--backbone",
            "resnet50",
            "--bert_enc_num",
            "12",
            "--detr_enc_num",
            "6",
            "--dataset",
            "flickr",
            "--max_query_len",
            str(args.max_query_len),
            "--bert_model",
            text_pin["snapshot_path"],
            "--output_dir",
            str(output),
            "--split_root",
            str(Path(args.split_root).resolve()),
            "--data_root",
            str(Path(args.data_root).resolve()),
            "--epochs",
            str(args.epochs),
            "--lr_drop",
            str(args.lr_drop),
            "--resume",
            str(Path(args.resume).resolve()),
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
        ]
    )
    train_args.distributed = False
    train_args.world_size = 1
    train_args.dist_url = "env://"
    transvg_train.main(train_args)


if __name__ == "__main__":
    main()
