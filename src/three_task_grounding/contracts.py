from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLITS = ("train", "val", "eval")

OutputMode = Literal["variable_set", "one_box"]
QueryMode = Literal["raw_phrase", "anatomy_name", "device_claim"]


@dataclass(frozen=True)
class ProtocolSpec:
    key: str
    display_name: str
    source_dataset: str
    source_root: Path
    source_pattern: str
    query_mode: QueryMode
    target_semantics: str
    output_mode: OutputMode
    expected_source_rows: dict[str, int]
    expected_groups: dict[str, int]
    primary_metrics: tuple[str, ...]
    query_contract: str
    task_supervision_note: str

    def source_path(self, split: str) -> Path:
        if split not in SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        return self.source_root / self.source_pattern.format(split=split)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["source_root"] = str(self.source_root)
        return payload


PROTOCOLS: dict[str, ProtocolSpec] = {
    "mscxr_multibox_1444": ProtocolSpec(
        key="mscxr_multibox_1444",
        display_name="MS-CXR phrase grounding, variable-cardinality 1444 protocol",
        source_dataset="MS-CXR",
        source_root=PROJECT_ROOT / "training" / "ms_cxr_vfm_localizer_stage1_p10_p19" / "datasets",
        source_pattern="{split}.jsonl",
        query_mode="raw_phrase",
        target_semantics="MS-CXR phrase-grounding rectangles",
        output_mode="variable_set",
        expected_source_rows={"train": 998, "val": 166, "eval": 280},
        # Canonical finding-conditioned membership used by controlled 1444 runs.
        expected_groups={"train": 813, "val": 124, "eval": 220},
        primary_metrics=("coverage_mean_iou", "exact_union_iou", "set_f1_0_3", "set_f1_0_5"),
        query_contract=(
            "MS-CXR finding query plus raw localized phrase; finding specifies what pathology to localize "
            "and is supplied identically across train/val/eval"
        ),
        task_supervision_note="MS-CXR train localization supervision only in the task-isolated main arm",
    ),
    "imagenome_anatomy_10k": ProtocolSpec(
        key="imagenome_anatomy_10k",
        display_name="Chest ImaGenome anatomy object localization, balanced 10k protocol",
        source_dataset="Chest ImaGenome",
        source_root=PROJECT_ROOT / "training" / "imagenome_device_anatomy_10k_rule_smm_v1" / "datasets",
        source_pattern="anatomy_{split}.jsonl",
        query_mode="anatomy_name",
        target_semantics="Chest ImaGenome anatomy object rectangle; not a lesion box",
        output_mode="one_box",
        expected_source_rows={"train": 6984, "val": 1008, "eval": 1998},
        expected_groups={"train": 6984, "val": 1008, "eval": 1998},
        primary_metrics=("mean_iou", "hit_0_3", "hit_0_5"),
        query_contract="the anatomy name is the explicit task query and is therefore allowed",
        task_supervision_note="ImaGenome anatomy supervision is used only for this target task by default",
    ),
    "imagenome_device_weak_10k": ProtocolSpec(
        key="imagenome_device_weak_10k",
        display_name="Chest ImaGenome device-linked weak-region localization, balanced 10k protocol",
        source_dataset="Chest ImaGenome",
        source_root=PROJECT_ROOT / "training" / "imagenome_device_anatomy_10k_rule_smm_v1" / "datasets",
        source_pattern="device_{split}.jsonl",
        query_mode="device_claim",
        target_semantics="device-linked anatomy/landmark weak rectangle; not an exact device contour",
        output_mode="one_box",
        expected_source_rows={"train": 6996, "val": 1002, "eval": 1998},
        expected_groups={"train": 6996, "val": 1002, "eval": 1998},
        primary_metrics=("mean_iou", "hit_0_3", "hit_0_5"),
        query_contract=(
            "normalized device name plus raw report claim; bbox_name_reference and object_name_reference "
            "are debug/label fields and are forbidden at inference"
        ),
        task_supervision_note="ImaGenome device-linked supervision is used only for this target task by default",
    ),
}


def get_protocol(key: str) -> ProtocolSpec:
    try:
        return PROTOCOLS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown protocol {key!r}; choose from {sorted(PROTOCOLS)}") from exc
