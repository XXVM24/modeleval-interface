"""Dataset quality checks and auto-cleaning for QA pair lists."""

import csv
import json
import re
import tempfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .runner import QAPair

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Control characters excluding tab (\x09), newline (\x0a), carriage return (\x0d)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Unicode replacement character (signals bad decoding)
_REPLACEMENT_CHAR = "\ufffd"
# Null byte
_NULL_BYTE = "\x00"


def _has_garbled(text: str) -> bool:
    """True if text contains replacement chars, null bytes, or control chars."""
    if not text:
        return False
    return (
        _REPLACEMENT_CHAR in text
        or _NULL_BYTE in text
        or bool(_CTRL_RE.search(text))
    )


def _is_meaningless(text: str) -> bool:
    """True if text is non-empty but contains zero alphanumeric characters."""
    if not text or not text.strip():
        return False
    return not any(c.isalnum() for c in text)


def _has_excess_whitespace(text: str) -> bool:
    """True if text has leading/trailing whitespace or internal runs of 2+ spaces."""
    if not text:
        return False
    return (
        text != text.strip()
        or bool(re.search(r"[ \t]{2,}", text))
    )


def _clean_text(text: str) -> Tuple[str, List[str]]:
    """
    Apply all cleaning transformations.  Returns (cleaned, [action_labels]).
    """
    original = text
    actions: List[str] = []

    # 1. Remove null bytes
    if _NULL_BYTE in text:
        text = text.replace(_NULL_BYTE, "")
        actions.append("removed null bytes")

    # 2. Remove replacement chars
    if _REPLACEMENT_CHAR in text:
        text = text.replace(_REPLACEMENT_CHAR, "")
        actions.append("removed replacement chars (\\ufffd)")

    # 3. Remove control characters (keep \t \n \r)
    cleaned = _CTRL_RE.sub("", text)
    if cleaned != text:
        text = cleaned
        actions.append("removed control characters")

    # 4. Strip leading/trailing whitespace
    stripped = text.strip()
    if stripped != text:
        text = stripped
        actions.append("stripped leading/trailing whitespace")

    # 5. Collapse internal whitespace runs
    collapsed = re.sub(r"[ \t]{2,}", " ", text)
    if collapsed != text:
        text = collapsed
        actions.append("collapsed internal whitespace")

    return text, actions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(pairs: List[QAPair]) -> dict:
    """Scan pairs for quality issues and return counts + row indices per issue category."""
    issues: dict = {
        "garbled":     [],
        "meaningless": [],
        "missing_q":   [],
        "missing_ref": [],
        "whitespace":  [],
        "duplicates":  [],
    }

    seen: dict = {}  # (question, reference) -> first row index

    for i, p in enumerate(pairs):
        q = p.question or ""
        r = p.reference_answer or ""
        row_issues = set()

        # Missing fields (check on raw values)
        if not q.strip():
            issues["missing_q"].append(i)
            row_issues.add("missing_q")
        if not r.strip():
            issues["missing_ref"].append(i)
            row_issues.add("missing_ref")

        # Garbled chars
        if _has_garbled(q) or _has_garbled(r):
            issues["garbled"].append(i)
            row_issues.add("garbled")

        # Meaningless content (non-empty but no alphanumeric chars)
        if _is_meaningless(q) or _is_meaningless(r):
            issues["meaningless"].append(i)
            row_issues.add("meaningless")

        # Excess whitespace
        if _has_excess_whitespace(q) or _has_excess_whitespace(r):
            issues["whitespace"].append(i)
            row_issues.add("whitespace")

        # Duplicates
        key = (q.strip(), r.strip())
        if key in seen:
            issues["duplicates"].append(i)
            row_issues.add("duplicates")
        else:
            seen[key] = i

    total = len(pairs)
    all_affected = set()
    for idx_list in issues.values():
        all_affected.update(idx_list)
    clean_count = total - len(all_affected)

    # Build summary lines for report table
    _LABELS = [
        ("garbled",     "Garbled characters"),
        ("meaningless", "Meaningless content"),
        ("missing_q",   "Missing question"),
        ("missing_ref", "Missing reference answer"),
        ("whitespace",  "Excess whitespace"),
        ("duplicates",  "Duplicate rows"),
    ]
    summary_lines = []
    for key, label in _LABELS:
        idxs = issues[key]
        count = len(idxs)
        if count:
            shown = idxs[:10]
            indices_str = ", ".join(str(i + 1) for i in shown)
            if len(idxs) > 10:
                indices_str += f", … (+{len(idxs) - 10} more)"
        else:
            indices_str = "—"
        summary_lines.append((label, count, indices_str))

    return {
        "total": total,
        "issues": issues,
        "clean_count": clean_count,
        "summary_lines": summary_lines,
    }


def clean(pairs: List[QAPair]) -> Tuple[List[QAPair], List[dict]]:
    """Clean all pairs and return (cleaned_pairs, change_log). Drops empty and duplicate rows."""
    cleaned_pairs: List[QAPair] = []
    change_log: List[dict] = []

    seen: set = set()  # for deduplication
    dropped_empty = 0
    dropped_dup = 0

    for i, p in enumerate(pairs):
        q_orig = p.question or ""
        r_orig = p.reference_answer or ""

        q_clean, q_actions = _clean_text(q_orig)
        r_clean, r_actions = _clean_text(r_orig)

        # Log changes
        if q_actions:
            change_log.append({
                "row": i + 1,
                "field": "question",
                "before": q_orig,
                "after": q_clean,
                "actions": q_actions,
            })
        if r_actions:
            change_log.append({
                "row": i + 1,
                "field": "reference_answer",
                "before": r_orig,
                "after": r_clean,
                "actions": r_actions,
            })

        # Drop rows that are completely empty after cleaning
        if not q_clean.strip() and not r_clean.strip():
            dropped_empty += 1
            continue

        # Deduplicate
        key = (q_clean.strip(), r_clean.strip())
        if key in seen:
            dropped_dup += 1
            continue
        seen.add(key)

        cleaned_pairs.append(QAPair(question=q_clean, reference_answer=r_clean))

    return cleaned_pairs, change_log


def pairs_to_dataframe(pairs: List[QAPair]) -> pd.DataFrame:
    """Convert QAPairs to a two-column DataFrame for Gradio preview."""
    return pd.DataFrame(
        [{"#": i + 1, "question": p.question, "reference_answer": p.reference_answer}
         for i, p in enumerate(pairs)]
    )


def save_cleaned(pairs: List[QAPair], original_path: str) -> str:
    """
    Save cleaned pairs to a temp file in the same format as the original.
    Returns the output file path.
    """
    suffix = Path(original_path).suffix.lower()
    stem = Path(original_path).stem

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=suffix,
        prefix=f"{stem}_cleaned_",
        delete=False,
        encoding="utf-8",
        newline="" if suffix == ".csv" else "\n",
    )

    if suffix in (".json",):
        data = [{"question": p.question, "reference_answer": p.reference_answer}
                for p in pairs]
        json.dump(data, tmp, ensure_ascii=False, indent=2)

    elif suffix == ".jsonl":
        for p in pairs:
            json.dump({"question": p.question, "reference_answer": p.reference_answer},
                      tmp, ensure_ascii=False)
            tmp.write("\n")

    elif suffix == ".csv":
        writer = csv.DictWriter(tmp, fieldnames=["question", "reference_answer"])
        writer.writeheader()
        for p in pairs:
            writer.writerow({"question": p.question, "reference_answer": p.reference_answer})

    elif suffix == ".txt":
        for p in pairs:
            tmp.write(f"{p.question}\t{p.reference_answer}\n")

    else:
        # Fallback: JSON
        data = [{"question": p.question, "reference_answer": p.reference_answer}
                for p in pairs]
        json.dump(data, tmp, ensure_ascii=False, indent=2)

    tmp.close()
    return tmp.name
