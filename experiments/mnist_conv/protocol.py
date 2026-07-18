"""Central approval gate for executable MNIST Conv protocols.

Schema validation deliberately permits ``category="final"`` so a proposed
configuration can be reviewed, fingerprinted, and expanded into a plan before
approval.  Execution is a separate privilege and fails closed here.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from .identity import sha256_json

if TYPE_CHECKING:  # Avoid importing the schema solely to evaluate the gate.
    from .specs import RunSpec


PROTOCOL_CONTRACT_SCHEMA_VERSION = "mnist-conv-protocol-contract/v1"


# Intentionally empty while operational T/K and the later training protocol
# remain pending. Each reviewed protocol id maps to the complete set of exact
# resolved-run contract fingerprints it authorizes. This supports a multi-case,
# multi-seed final sweep without allowing the id alone to approve config drift.
APPROVED_FINAL_PROTOCOL_CONTRACTS: Mapping[str, frozenset[str]] = MappingProxyType({})

# Compatibility/readability view. Authorization must always consult the
# contract mapping above, never this id-only view.
APPROVED_FINAL_PROTOCOL_IDS: frozenset[str] = frozenset(
    APPROVED_FINAL_PROTOCOL_CONTRACTS
)


class ProtocolExecutionError(RuntimeError):
    """Raised when a valid run is not approved for numerical execution."""


def protocol_contract_fingerprint(spec: "RunSpec") -> str:
    """Fingerprint one exact normalized scientific run contract.

    ``RunSpec.identity_payload`` already excludes display labels and portable
    checkpoint paths. The registry key supplies the protocol id, while the
    final-category gate is enforced separately, so neither is duplicated in
    the scientific payload. Seed and replicate id remain included.
    """

    payload = spec.identity_payload()
    payload.pop("protocol_id")
    payload.pop("category")
    return "protocol_contract_" + sha256_json(
        {
            "schema_version": PROTOCOL_CONTRACT_SCHEMA_VERSION,
            "scientific_run": payload,
        }
    )


def final_protocol_approval_reason(spec: "RunSpec") -> str | None:
    """Return the stable collector reason for a missing approval binding."""

    protocol_id = spec.data["protocol_id"]
    approved_contracts = APPROVED_FINAL_PROTOCOL_CONTRACTS.get(protocol_id)
    if approved_contracts is None:
        return "protocol_not_approved"
    if protocol_contract_fingerprint(spec) not in approved_contracts:
        return "protocol_contract_mismatch"
    return None


def ensure_run_executable(spec: "RunSpec") -> None:
    """Fail closed unless a final run's exact contract is approved.

    Diagnostic runs remain executable because their category already prevents
    paper eligibility. Final specifications remain valid planning inputs, but
    their fingerprint must appear under their protocol id in
    :data:`APPROVED_FINAL_PROTOCOL_CONTRACTS` before any backend work.
    """

    value = spec.data
    if value["category"] != "final":
        return
    protocol_id = value["protocol_id"]
    reason = final_protocol_approval_reason(spec)
    if reason is not None:
        fingerprint = protocol_contract_fingerprint(spec)
        raise ProtocolExecutionError(
            "Expected a final MNIST Conv run to match an exact immutable scientific contract "
            "registered under its explicitly approved protocol id. "
            f"Provided protocol_id={protocol_id!r}, contract_fingerprint={fingerprint!r}, "
            f"approval_reason={reason!r}, approved_protocol_ids="
            f"{sorted(APPROVED_FINAL_PROTOCOL_CONTRACTS)!r}. "
            "The configuration may be validated and planned, but numerical execution is blocked."
        )
