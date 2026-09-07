from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import pandas as pd


SIDE_CUE_TYPES = {
    "bilateral",
    "both_sides",
    "right_and_left",
    "left_and_right",
    "bibasilar",
    "both_bases",
    "plural_effusions",
    "plural_pneumothoraces",
    "upper_lobes",
    "lower_lobes",
}

DIFFUSE_CUE_TYPES = {"diffuse", "multifocal", "widespread", "scattered", "patchy"}


def norm_text(text: str | None) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _base_target(finding: str, laterality: str = "unknown", vertical: str = "unknown") -> dict[str, str]:
    return {"finding": finding or "", "laterality": laterality or "unknown", "vertical": vertical or "unknown"}


def _unique_targets(targets: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for target in targets:
        key = (
            str(target.get("finding", "")),
            str(target.get("laterality", "unknown")),
            str(target.get("vertical", "unknown")),
        )
        if key not in seen:
            out.append(target)
            seen.add(key)
    return out


def _has_unilateral_only(text: str, laterality: str) -> bool:
    if re.search(r"\b(bilateral|bilaterally|both|right\s+and\s+left|left\s+and\s+right)\b", text):
        return False
    if re.search(r"\bright\s+(greater|more)\s+than\s+left\b", text):
        return False
    if re.search(r"\bleft\s+(greater|more)\s+than\s+right\b", text):
        return False
    if laterality in {"right", "left"}:
        return True
    return bool(re.search(r"\b(right|left)\b", text))


def parse_multibox_cue(
    phrase: str | None,
    *,
    finding: str = "",
    laterality: str = "unknown",
    vertical: str = "unknown",
    rule_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse phrase-level cues that justify predicting more than one box.

    The parser uses only phrase/rule-context text.  It never uses gold box
    count and is therefore safe for val/test inference after val-time tuning.
    """

    text = norm_text(phrase)
    lat = str((rule_context or {}).get("laterality", laterality) or "unknown").lower()
    vert = str((rule_context or {}).get("vertical", vertical) or "unknown").lower()
    finding = str(finding or (rule_context or {}).get("finding", ""))

    cue_type = "none"
    cues: list[str] = []
    target_qs: list[dict[str, str]] = []
    k_hint = 1

    explicit_unilateral = _has_unilateral_only(text, lat)

    def add_side_targets(cue: str, v: str | None = None) -> None:
        nonlocal cue_type, k_hint
        cue_type = cue_type if cue_type != "none" else cue
        cues.append(cue)
        vv = v or vert or "unknown"
        target_qs.extend([
            _base_target(finding, "right", vv),
            _base_target(finding, "left", vv),
        ])
        k_hint = max(k_hint, 2)

    if re.search(r"\bbilateral(ly)?\b", text):
        add_side_targets("bilateral")
    elif re.search(r"\bboth\b", text):
        add_side_targets("both_sides")
    elif re.search(r"\bright\s+and\s+left\b", text):
        add_side_targets("right_and_left")
    elif re.search(r"\bleft\s+and\s+right\b", text):
        add_side_targets("left_and_right")

    if re.search(r"\b(bibasilar|bibasal)\b", text):
        add_side_targets("bibasilar", "basal")
    elif re.search(r"\b(both\s+bases|lung\s+bases)\b", text):
        add_side_targets("both_bases", "basal")

    if re.search(r"\bupper\s+lobes\b", text):
        add_side_targets("upper_lobes", "upper")
    if re.search(r"\blower\s+lobes\b", text):
        add_side_targets("lower_lobes", "lower")

    if re.search(r"\bpleural\s+effusions\b|\beffusions\b", text) and not explicit_unilateral:
        add_side_targets("plural_effusions")
    if re.search(r"\bpneumothoraces\b", text) and not explicit_unilateral:
        add_side_targets("plural_pneumothoraces")

    if re.search(r"\bdiffuse\b", text):
        cue_type = cue_type if cue_type != "none" else "diffuse"
        cues.append("diffuse")
        k_hint = max(k_hint, 3)
    if re.search(r"\b(multifocal|multisegmental|multilobar|multiple)\b", text):
        cue_type = cue_type if cue_type != "none" else "multifocal"
        cues.append("multifocal")
        k_hint = max(k_hint, 3)
    if re.search(r"\b(widespread|extensive)\b", text):
        cue_type = cue_type if cue_type != "none" else "widespread"
        cues.append("widespread")
        k_hint = max(k_hint, 3)
    if re.search(r"\bscattered\b", text):
        cue_type = cue_type if cue_type != "none" else "scattered"
        cues.append("scattered")
        k_hint = max(k_hint, 2)
    if re.search(r"\bpatchy\b", text):
        cue_type = cue_type if cue_type != "none" else "patchy"
        cues.append("patchy")
        k_hint = max(k_hint, 2)

    if finding in {"Cardiomegaly"}:
        cues = []
        cue_type = "none"
        target_qs = []
        k_hint = 1

    target_qs = _unique_targets(target_qs)
    k_hint = min(max(int(k_hint), len(target_qs), 1), 4)
    has_multi = bool(cues)
    if not has_multi:
        target_qs = [_base_target(finding, lat if lat in {"left", "right"} else "unknown", vert)]

    source = "none"
    if has_multi:
        source = "phrase"
        if lat not in {"unknown", "none", ""} or vert not in {"unknown", "none", ""}:
            source = "both"

    return {
        "has_multi_cue": has_multi,
        "multi_cue_type": cue_type,
        "cue_text": ";".join(dict.fromkeys(cues)) if cues else "single_or_unspecified",
        "k_hint": k_hint if has_multi else 1,
        "target_qs": target_qs,
        "cue_source": source,
        "cue_debug_string": f"type={cue_type}; cues={';'.join(cues) if cues else 'none'}; k={k_hint}; targets={json.dumps(target_qs)}",
    }


def parse_precision_multibox_cue(
    phrase: str | None,
    *,
    finding: str = "",
    laterality: str = "unknown",
    vertical: str = "unknown",
    rule_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use only high-precision phrase constructions as hard multi-box cues.

    Distribution words such as ``patchy``, ``diffuse``, and ``extensive`` can
    describe one broad annotated rectangle. They remain scoring context, but
    do not alone force another output box. This parser is opt-in so the
    historical decoder remains reproducible.
    """
    text = norm_text(phrase)
    base = parse_multibox_cue(
        phrase,
        finding=finding,
        laterality=laterality,
        vertical=vertical,
        rule_context=rule_context,
    )
    finding_name = str(finding or (rule_context or {}).get("finding", ""))
    lat = str((rule_context or {}).get("laterality", laterality) or "unknown").lower()
    vert = str((rule_context or {}).get("vertical", vertical) or "unknown").lower()

    has_side_pair = bool(re.search(r"\b(bilateral|bilaterally|both|right\s+and\s+left|left\s+and\s+right)\b", text))
    has_basal_pair = bool(re.search(r"\b(bibasilar|bibasal|both\s+bases|lung\s+bases)\b", text))
    has_plural_effusions = bool(
        re.search(r"\bpleural\s+effusions\b|\beffusions\b", text)
        and not _has_unilateral_only(text, lat)
    )
    has_multifocal = bool(re.search(r"\bmultifocal\b", text))

    if has_side_pair or has_basal_pair or has_plural_effusions:
        base["has_multi_cue"] = True
        base["k_hint"] = 2
        base["cue_debug_string"] = f"precision:{base.get('cue_debug_string', '')}; hard_floor=2"
        return base
    if has_multifocal:
        return {
            "has_multi_cue": True,
            "multi_cue_type": "multifocal_precision",
            "cue_text": "multifocal_precision",
            "k_hint": 2,
            "target_qs": [],
            "cue_source": "phrase",
            "cue_debug_string": "precision:multifocal; hard_floor=2; targets=[]",
        }
    return {
        "has_multi_cue": False,
        "multi_cue_type": "none",
        "cue_text": "single_or_unspecified",
        "k_hint": 1,
        "target_qs": [_base_target(finding_name, lat if lat in {"left", "right"} else "unknown", vert)],
        "cue_source": "none",
        "cue_debug_string": "precision:no high-precision multi-region cue",
    }


def cue_info_from_groups(groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for gid, g in groups.items():
        out[str(gid)] = parse_multibox_cue(
            g.get("claim_sentence", ""),
            finding=str(g.get("finding", "")),
            laterality=str(g.get("laterality", "unknown")),
            vertical=str(g.get("vertical", "unknown")),
        )
    return out


def cue_info_from_groups_precision(groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return opt-in high-precision cues for a group dictionary."""
    out: dict[str, dict[str, Any]] = {}
    for gid, g in groups.items():
        out[str(gid)] = parse_precision_multibox_cue(
            g.get("claim_sentence", ""),
            finding=str(g.get("finding", "")),
            laterality=str(g.get("laterality", "unknown")),
            vertical=str(g.get("vertical", "unknown")),
        )
    return out


def cue_info_from_candidate_table(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if df.empty:
        return out
    for qid, part in df.groupby("query_id", sort=False):
        row = part.iloc[0]
        out[str(qid)] = parse_multibox_cue(
            row.get("claim_sentence", row.get("phrase", "")),
            finding=str(row.get("finding", "")),
            laterality=str(row.get("laterality", "unknown")),
            vertical=str(row.get("vertical", "unknown")),
        )
    return out


def cue_audit_frame(groups: dict[str, dict[str, Any]], cue_info: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gid, g in groups.items():
        cue = cue_info.get(str(gid), parse_multibox_cue(g.get("claim_sentence", ""), finding=g.get("finding", "")))
        rows.append(
            {
                "query_id": gid,
                "group_id": gid,
                "phrase": g.get("claim_sentence", ""),
                "finding": g.get("finding", ""),
                "laterality": cue["target_qs"][0].get("laterality", "unknown") if cue.get("target_qs") else "unknown",
                "vertical": cue["target_qs"][0].get("vertical", "unknown") if cue.get("target_qs") else "unknown",
                "gold_count": len(g.get("gt_boxes", [])),
                "has_multi_cue": bool(cue.get("has_multi_cue", False)),
                "multi_cue_type": cue.get("multi_cue_type", "none"),
                "k_hint": int(cue.get("k_hint", 1)),
                "target_qs": json.dumps(cue.get("target_qs", []), ensure_ascii=False),
                "cue_source": cue.get("cue_source", "none"),
                "cue_debug_string": cue.get("cue_debug_string", ""),
            }
        )
    return pd.DataFrame(rows)


def cue_type_counts(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    return (
        audit.groupby(["has_multi_cue", "multi_cue_type"], dropna=False)
        .size()
        .reset_index(name="n_queries")
        .sort_values(["has_multi_cue", "n_queries"], ascending=[False, False])
    )


def cue_gold_count_distribution(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for cue_type, part in audit.groupby("multi_cue_type", dropna=False):
        counts = Counter(int(x) for x in part["gold_count"].fillna(0))
        row: dict[str, Any] = {"multi_cue_type": cue_type, "n_queries": int(len(part))}
        for k in sorted(counts):
            row[f"gold_count_{k}"] = int(counts[k])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_queries", ascending=False)
