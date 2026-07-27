"""Cross-worktree experiment orchestration."""

from campaigns.schema import (
    CampaignManifestError,
    CampaignSpec,
    load_campaign_manifest,
)


def run_campaign(*args, **kwargs):
    """Lazily load the subprocess runner for lightweight schema consumers."""

    from campaigns.runner import run_campaign as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "CampaignManifestError",
    "CampaignSpec",
    "load_campaign_manifest",
    "run_campaign",
]
