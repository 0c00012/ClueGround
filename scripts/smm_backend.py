#!/usr/bin/env python
"""Local SMM backend adapters for YoVIS quick experiments.

No MIMIC image/text data is sent to external APIs. If no local SMM backend is
available, callers must report the run as blocked instead of fabricating
performance metrics.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "smm_backend.yaml"

_HF_RUNNER = None


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def read_simple_yaml(path: Path = DEFAULT_CONFIG) -> Dict[str, object]:
    config: Dict[str, object] = {}
    if not path.exists():
        return config
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.lower() in {"true", "false"}:
            parsed: object = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value.strip('"').strip("'")
        config[key.strip()] = parsed
    return config


def detect_backend(config: Optional[Dict[str, object]] = None) -> Dict[str, str]:
    config = config or read_simple_yaml()
    requested = os.environ.get("YOVIS_SMM_BACKEND", str(config.get("backend", "auto"))).strip().lower()
    if requested == "stub_debug" and os.environ.get("ALLOW_STUB_DEBUG") == "1":
        return {"backend": "stub_debug", "model_name": "stub_debug_not_for_metrics", "available": "true"}

    cli = os.environ.get("YOVIS_SMM_CLI")
    if cli:
        exe = cli.split()[0]
        if shutil.which(exe) or Path(exe).exists():
            return {"backend": "local_cli", "model_name": cli, "available": "true"}
        return {"backend": "local_cli", "model_name": cli, "available": "false", "reason": "YOVIS_SMM_CLI executable not found"}

    if requested in {"auto", "local_hf"}:
        missing = [p for p in ["torch", "transformers", "qwen_vl_utils", "PIL"] if not package_available(p)]
        model_name = os.environ.get("YOVIS_HF_MODEL", str(config.get("model_name", ""))).strip()
        if not missing and model_name:
            return {
                "backend": "local_hf",
                "model_name": model_name,
                "available": "true",
                "max_new_tokens": str(config.get("max_new_tokens", 256)),
                "temperature": str(config.get("temperature", 0.0)),
                "top_p": str(config.get("top_p", 1.0)),
                "dtype": str(config.get("dtype", "auto")),
                "device": str(config.get("device", "auto")),
                "image_max_side": str(config.get("image_max_side", 1024)),
                "allow_multi_image": str(config.get("allow_multi_image", True)),
            }
        return {
            "backend": "local_hf",
            "model_name": model_name,
            "available": "false",
            "reason": "Missing packages for local_hf: " + ", ".join(missing) if missing else "No model_name configured",
        }

    for exe in ["ollama", "llama-cli"]:
        if shutil.which(exe):
            return {"backend": "local_cli", "model_name": exe, "available": "false", "reason": f"{exe} found, but no configured vision prompt wrapper/YOVIS_SMM_CLI"}

    return {
        "backend": "none",
        "model_name": "",
        "available": "false",
        "reason": "No local SMM backend detected. Configure local_hf packages/model or YOVIS_SMM_CLI.",
    }


def parse_json_from_text(text: str) -> tuple[Dict, bool, str]:
    text = str(text or "").strip()
    if not text:
        return {"prediction_status": "parse_error"}, False, "empty_output"
    try:
        return json.loads(text), True, ""
    except Exception as exc:
        direct_error = repr(exc)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), True, ""
        except Exception as exc:
            return {"prediction_status": "parse_error"}, False, f"json_recovery_failed: {exc!r}; direct={direct_error}"
    return {"prediction_status": "parse_error"}, False, f"json_object_not_found; direct={direct_error}"


def normalize_prediction(parsed: Dict, raw_output: str = "", parse_ok: bool = True, parse_error: str = "") -> Dict:
    status = str(parsed.get("prediction_status") or parsed.get("status") or parsed.get("answer") or "").lower()
    present = bool(parsed.get("prediction_present", False)) or status == "present"
    absent = bool(parsed.get("prediction_absent", False)) or status == "absent"
    uncertain = bool(parsed.get("prediction_uncertain", False)) or status in {"uncertain", "unsure", "review_needed", "parse_error"}
    if sum([present, absent, uncertain]) == 0:
        uncertain = True
        status = "uncertain"
    if present:
        status = "present"
        absent = False
        uncertain = False
    elif absent:
        status = "absent"
        present = False
        uncertain = False
    elif uncertain:
        status = status or "uncertain"
        present = False
        absent = False
    try:
        conf = float(parsed.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    return {
        "prediction_status": status,
        "prediction_present": present,
        "prediction_absent": absent,
        "prediction_uncertain": uncertain,
        "location_text": str(parsed.get("location_text") or parsed.get("location") or ""),
        "confidence": conf,
        "evidence_sentence": str(parsed.get("evidence_sentence") or parsed.get("evidence") or ""),
        "revision_reason": str(parsed.get("revision_reason") or parsed.get("reason") or ""),
        "raw_output": raw_output or json.dumps(parsed, ensure_ascii=False),
        "parse_ok": parse_ok,
        "parse_error": parse_error,
    }


def run_stub(task: Dict, method: str) -> Dict:
    # Debug only. Evaluation code refuses stub_debug metrics by default.
    return normalize_prediction(
        {
            "prediction_status": "uncertain",
            "prediction_present": False,
            "prediction_absent": False,
            "prediction_uncertain": True,
            "confidence": 0.0,
            "revision_reason": "stub_debug_not_for_metrics",
        }
    )


def run_local_cli(task: Dict, method: str, prompt: str, cli_cmd: str) -> Dict:
    # User-provided CLI wrapper must accept prompt on stdin and return JSON on stdout.
    # Image paths are embedded in the prompt. This never calls an external API by itself.
    start = time.time()
    proc = subprocess.run(
        cli_cmd,
        input=prompt,
        capture_output=True,
        text=True,
        shell=True,
        timeout=int(os.environ.get("YOVIS_SMM_TIMEOUT", "180")),
    )
    raw = proc.stdout.strip() if proc.stdout else proc.stderr.strip()
    parsed, parse_ok, parse_error = parse_json_from_text(raw)
    pred = normalize_prediction(parsed, raw, parse_ok, parse_error)
    pred["latency_sec"] = time.time() - start
    return pred


class LocalHFQwenRunner:
    def __init__(self, backend_info: Dict[str, str]):
        self.backend_info = backend_info
        self.model_name = backend_info["model_name"]
        self.max_new_tokens = int(float(backend_info.get("max_new_tokens", 256)))
        self.temperature = float(backend_info.get("temperature", 0.0))
        self.top_p = float(backend_info.get("top_p", 1.0))
        self.image_max_side = int(float(backend_info.get("image_max_side", 1024)))
        self.model = None
        self.processor = None
        self.torch = None
        self.process_vision_info = None
        self.device_label = ""
        self._load()

    def _load(self) -> None:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        except Exception:
            try:
                from transformers import AutoModelForImageTextToText as ModelClass
            except Exception:
                from transformers import AutoModelForVision2Seq as ModelClass

        self.torch = torch
        self.process_vision_info = process_vision_info
        dtype_cfg = self.backend_info.get("dtype", "auto")
        if dtype_cfg == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        elif dtype_cfg in {"bf16", "bfloat16"}:
            dtype = torch.bfloat16
        elif dtype_cfg in {"fp16", "float16"}:
            dtype = torch.float16
        else:
            dtype = "auto"

        kwargs = {"device_map": "auto"}
        if dtype != "auto":
            kwargs["torch_dtype"] = dtype
        self.model = ModelClass.from_pretrained(self.model_name, **kwargs)
        max_pixels = self.image_max_side * self.image_max_side
        self.processor = AutoProcessor.from_pretrained(self.model_name, max_pixels=max_pixels)
        if torch.cuda.is_available():
            self.device_label = torch.cuda.get_device_name(0)
        else:
            self.device_label = "cpu"

    def generate(self, prompt: str, image_paths: List[str]) -> str:
        torch = self.torch
        content = []
        for path in image_paths:
            if path and Path(path).exists():
                content.append({"type": "image", "image": str(path)})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        try:
            device = self.model.device
            inputs = inputs.to(device)
        except Exception:
            inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        gen_kwargs = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": self.temperature, "top_p": self.top_p})
        else:
            gen_kwargs.update({"do_sample": False})
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def get_hf_runner(backend_info: Dict[str, str]) -> LocalHFQwenRunner:
    global _HF_RUNNER
    if _HF_RUNNER is None:
        _HF_RUNNER = LocalHFQwenRunner(backend_info)
    return _HF_RUNNER


def run_local_hf(task: Dict, method: str, prompt: str, backend_info: Dict[str, str], image_paths: Optional[List[str]] = None) -> Dict:
    start = time.time()
    runner = get_hf_runner(backend_info)
    paths = image_paths or [task.get("image_path", "")]
    raw = runner.generate(prompt, paths)
    parsed, parse_ok, parse_error = parse_json_from_text(raw)
    pred = normalize_prediction(parsed, raw, parse_ok, parse_error)
    pred["latency_sec"] = time.time() - start
    return pred


def run_prediction(
    task: Dict,
    method: str,
    prompt: str,
    backend_info: Dict[str, str],
    image_paths: Optional[List[str]] = None,
) -> Dict:
    start = time.time()
    backend = backend_info["backend"]
    if backend == "stub_debug":
        pred = run_stub(task, method)
    elif backend == "local_cli":
        pred = run_local_cli(task, method, prompt, backend_info["model_name"])
    elif backend == "local_hf":
        pred = run_local_hf(task, method, prompt, backend_info, image_paths=image_paths)
    else:
        raise RuntimeError(backend_info.get("reason", "No SMM backend available"))
    pred.setdefault("latency_sec", time.time() - start)
    pred["backend"] = backend
    pred["model_name"] = backend_info.get("model_name", "")
    return pred
