from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_experimental_manifest_has_one_concise_presentation_per_entry() -> None:
    document = (REPO_ROOT / "docs" / "experimental_manifest.md").read_text(
        encoding="utf-8"
    )
    identifier = r"[a-z0-9][a-z0-9._-]*"
    begin_ids = re.findall(
        rf"^<!-- BEGIN EBL STUDY SUMMARY ({identifier}) -->$",
        document,
        re.MULTILINE,
    )
    end_ids = re.findall(
        rf"^<!-- END EBL STUDY SUMMARY ({identifier}) -->$",
        document,
        re.MULTILINE,
    )
    blocks = re.findall(
        rf"^<!-- BEGIN EBL STUDY SUMMARY ({identifier}) -->\n"
        rf"(.*?)\n<!-- END EBL STUDY SUMMARY \1 -->$",
        document,
        re.MULTILINE | re.DOTALL,
    )

    assert begin_ids
    assert begin_ids == end_ids
    assert len(blocks) == len(begin_ids)
    assert len(begin_ids) == len(set(begin_ids))
    assert document.count("<details>") == len(begin_ids)
    assert document.count("</details>") == len(begin_ids)

    for study_id, block in blocks:
        heading = re.search(
            rf"^### \[{re.escape(study_id)}\]\(([^)]+)\)$",
            block,
            re.MULTILINE,
        )
        interpretations = re.findall(
            r"^- \*\*Interpretation:\*\* (.+)$",
            block,
            re.MULTILINE,
        )
        assert heading is not None
        assert heading.group(1).startswith("../")
        assert len(interpretations) == 1
        assert len(interpretations[0]) <= 600
        assert "- **Outcome:**" in block
        assert "- **Details:**" in block


def test_current_state_is_synthesis_not_an_activity_ledger() -> None:
    document = (REPO_ROOT / "docs" / "current_state.md").read_text(
        encoding="utf-8"
    )

    assert "## Central research question" in document
    assert "## Current synthesis" in document
    assert "## Current priorities" in document
    assert "## Open questions" in document
    assert "## Personal notes" in document
    assert "BEGIN AUTOMATIC ACTIVE SIMULATIONS" not in document
    assert "Active native runs" not in document
