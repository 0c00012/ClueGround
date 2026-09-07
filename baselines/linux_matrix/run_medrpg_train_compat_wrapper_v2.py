from __future__ import annotations

import argparse
import json
import os
import random
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--detr-model", default="")
    parser.add_argument("--epochs", default="20")
    parser.add_argument("--batch-size", default="8")
    parser.add_argument("--seed", default="42")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", default="0")
    parser.add_argument("--lr", default="0.00005")
    parser.add_argument("--lr-bert", default="0.00001")
    parser.add_argument("--lr-drop", type=int, default=60)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_log = output / "checkpoint_save_events.jsonl"
    enable_offline_hf()
    text_pin = pin_record("bert-base-uncased", require_main_ref=True)
    (output / "hf_text_encoder_pin.json").write_text(
        json.dumps(text_pin, indent=2), encoding="utf-8"
    )

    os.chdir(repo)
    sys.path.insert(0, str(repo))

    import torch
    import torch.distributed as dist
    import torch.nn.modules.linear as linear_mod
    import torchvision.ops as ops
    import torchvision.ops.misc as misc

    # MedRPG targets torch 1.7 / torchvision 0.8. Keep the original repo
    # untouched and apply compatibility shims only in this wrapper process.
    ops._new_empty_tensor = lambda input, shape: input.new_empty(shape)
    misc._output_size = (
        lambda dim, input, size, scale_factor:
        list(size) if size is not None else [int(input.shape[-dim + i] * scale_factor) for i in range(dim)]
    )
    linear_mod._LinearWithBias = torch.nn.Linear
    dist.all_reduce = lambda tensor, *a, **kw: tensor

    old_load = torch.load
    torch.load = lambda *a, **kw: old_load(*a, **({"weights_only": False} | kw))
    install_legacy_bert_safetensors_loader(text_pin["snapshot_path"])

    old_dumps = json.dumps
    json.dumps = (
        lambda obj, *a, **kw:
        old_dumps(obj, *a, default=lambda x: float(x.item()) if hasattr(x, "item") else str(x), **kw)
    )

    med_args = [
        "train.py",
        "--model_name", "TransVG_ca",
        "--seed", str(args.seed),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--lr_bert", str(args.lr_bert),
        "--lr_visu_cnn", "0.00001",
        "--lr_visu_tra", "0.00001",
        "--aug_crop",
        "--aug_scale",
        "--aug_translate",
        "--backbone", "resnet50",
        "--bert_enc_num", "12",
        "--detr_enc_num", "6",
        "--max_query_len", "20",
        "--bert_model", text_pin["snapshot_path"],
        "--CAsampleType", "random",
        "--CAsampleNum", "5",
        "--CAlossWeightBase", "0.05",
        "--CATextPoolType", "cls",
        "--CATemperature", "0.1",
        "--CAMode", "max_image_lcpTriple",
        "--dataset", "MS_CXR",
        "--output_dir", str(output),
        "--split_root", str(Path(args.split_root).resolve()),
        "--data_root", str(Path(args.data_root).resolve()),
        "--epochs", str(args.epochs),
        "--lr_drop", str(args.lr_drop),
        "--device", args.device,
        "--num_workers", str(args.num_workers),
    ]
    if args.resume:
        med_args += ["--resume", str(Path(args.resume).resolve()), "--resume_model_only"]
    elif args.detr_model:
        med_args += ["--detr_model", str(Path(args.detr_model).resolve())]

    sys.argv = med_args
    import train as med_train
    import datasets.data_loader as data_loader
    import utils.box_utils as box_utils
    import utils.misc as misc_utils

    def append_save_event(event: dict) -> None:
        event = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **event}
        with save_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def remap_checkpoint_path(path_like) -> Path:
        dst = Path(path_like).resolve()
        name_map = {
            "checkpoint.pth": "last.pth",
            "best_accu_checkpoint.pth": "best_accu.pth",
            "best_miou_checkpoint.pth": "best.pth",
        }
        if dst.name in name_map:
            dst = dst.with_name(name_map[dst.name])
        return dst

    def slim_checkpoint(obj, dst: Path, fallback: bool = False):
        if not isinstance(obj, dict) or "model" not in obj:
            return obj
        is_eval_checkpoint = dst.name.startswith("best") or dst.name == "last.pth"
        if is_eval_checkpoint or fallback:
            keep = {
                "model": obj["model"],
                "epoch": obj.get("epoch"),
                "args": obj.get("args"),
                "note": "Model-only checkpoint saved by run_medrpg_train_compat_wrapper_v2.py",
            }
            for key in ["val_accu", "val_miou", "best_accu", "best_miou"]:
                if key in obj:
                    keep[key] = obj[key]
            return keep
        return obj

    def atomic_torch_save(obj, dst: Path) -> tuple[bool, str]:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f"{dst.stem}.tmp_{os.getpid()}_{time.time_ns()}{dst.suffix}")
        try:
            torch.save(obj, tmp, _use_new_zipfile_serialization=False)
            last_error = None
            for _ in range(5):
                try:
                    os.replace(tmp, dst)
                    last_error = None
                    break
                except PermissionError as exc:
                    last_error = exc
                    time.sleep(0.5)
            if last_error is not None:
                raise last_error
            loaded = torch.load(dst, map_location="cpu")
            if isinstance(loaded, dict) and "model" not in loaded and "model_state_dict" not in loaded:
                return False, "load_test_missing_model_key"
            return True, "ok"
        except Exception as exc:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            return False, f"{type(exc).__name__}: {exc}"

    def safe_save_on_master(obj, path_like, *save_args, **save_kwargs):
        if not misc_utils.is_main_process():
            return
        dst = remap_checkpoint_path(path_like)
        epoch = obj.get("epoch") if isinstance(obj, dict) else None
        versioned_best = dst.with_name(f"best_epoch{int(epoch):04d}.pth") if dst.name == "best.pth" and epoch is not None else None
        slim = slim_checkpoint(obj, dst, fallback=False)
        if versioned_best is not None:
            ok_v, msg_v = atomic_torch_save(slim, versioned_best)
            append_save_event({
                "requested_path": str(path_like),
                "saved_path": str(versioned_best),
                "status": "versioned_best_ok" if ok_v else "versioned_best_failed",
                "error": None if ok_v else msg_v,
                "size_mb": round(versioned_best.stat().st_size / (1024 * 1024), 2) if versioned_best.exists() else None,
            })
        ok, msg = atomic_torch_save(slim, dst)
        if not ok and isinstance(obj, dict) and "model" in obj:
            fallback_dst = dst.with_name(f"{dst.stem}.model_only{dst.suffix}")
            ok2, msg2 = atomic_torch_save(slim_checkpoint(obj, fallback_dst, fallback=True), fallback_dst)
            append_save_event({
                "requested_path": str(path_like),
                "saved_path": str(fallback_dst if ok2 else dst),
                "status": "fallback_model_only_ok" if ok2 else "failed",
                "primary_error": msg,
                "fallback_error": None if ok2 else msg2,
            })
            if ok2 and dst.name == "best.pth":
                try:
                    os.replace(fallback_dst, dst)
                except OSError as exc:
                    append_save_event({"status": "best_replace_failed", "error": str(exc), "path": str(dst)})
            return
        append_save_event({
            "requested_path": str(path_like),
            "saved_path": str(dst),
            "status": "ok" if ok else "failed",
            "error": None if ok else msg,
            "size_mb": round(dst.stat().st_size / (1024 * 1024), 2) if dst.exists() else None,
        })
        if ok and dst.name == "best.pth" and versioned_best is not None:
            for stale in dst.parent.glob("best_epoch*.pth"):
                if stale != versioned_best:
                    try:
                        stale.unlink()
                    except OSError as exc:
                        append_save_event({"status": "stale_best_cleanup_failed", "path": str(stale), "error": str(exc)})

    misc_utils.save_on_master = safe_save_on_master
    med_train.utils.save_on_master = safe_save_on_master

    def safe_sample_neg_bbox(box, CAsampleType, CAsampleNum, category=0, w=640, h=640):
        # Original MedRPG passes torch scalar tensors into random.randint,
        # which fails on recent Python. Convert only those bounds to Python
        # integers while preserving the official RNGs and rejection loop.
        assert CAsampleType in ["random", "attention", "crossImage", "crossBatch"]
        box = box.detach().cpu().float()
        ori_w = box[2] - box[0]
        ori_h = box[3] - box[1]
        neg_boxes = []
        attempts = 0
        while len(neg_boxes) < int(CAsampleNum):
            attempts += 1
            if CAsampleType != "random":
                raise NotImplementedError(f"Official MedRPG sampler does not implement {CAsampleType}")
            x_neg = torch.randint(1, int(w), (1,))
            y_neg = torch.randint(1, int(h), (1,))
            width_delta = random.randint(
                int(torch.round(-ori_w * 0.1).item()),
                int(torch.round(ori_w * 0.1).item()),
            )
            height_delta = random.randint(
                int(torch.round(-ori_h * 0.1).item()),
                int(torch.round(ori_h * 0.1).item()),
            )
            width_neg = ori_w + width_delta
            height_neg = ori_h + height_delta
            neg = torch.zeros(4)
            neg[0] = x_neg - 0.5 * width_neg
            neg[1] = y_neg - 0.5 * height_neg
            neg[2] = x_neg + 0.5 * width_neg
            neg[3] = y_neg + 0.5 * height_neg
            neg = torch.round(neg)
            if neg[0] < 0 or neg[1] < 0 or neg[2] >= w or neg[3] >= h:
                continue
            iou, _ = box_utils.box_iou(box.unsqueeze(0), neg.unsqueeze(0))
            if float(iou.item()) > 0.25 and attempts < 300:
                continue
            neg_boxes.append(neg)
        return neg_boxes

    box_utils.sampleNegBBox = safe_sample_neg_bbox
    data_loader.sampleNegBBox = safe_sample_neg_bbox

    med_train.main(med_train.get_args_parser().parse_args())


if __name__ == "__main__":
    main()
