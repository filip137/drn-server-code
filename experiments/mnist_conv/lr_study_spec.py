"""Frozen declarative contracts for the Conv1/Conv2/Conv3 SGD LR studies.

The study schema is intentionally separate from the run and sweep schemas.
It describes the scientific choices shared by the probe, range, candidate,
and selection stages; executor choice and filesystem placement are not part
of the scientific identity.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .identity import canonical_json_bytes, sha256_json
from .specs import SpecValidationError


LR_STUDY_SCHEMA_VERSION = "mnist-conv-lr-study/v1"
LR_RESCUE_STUDY_SCHEMA_VERSION = "mnist-conv-lr-study/v2"
LR_RELATIVE_RHO_STUDY_SCHEMA_VERSION = "mnist-conv-lr-study/v3"
LR_LAYERWISE_RHO_STUDY_SCHEMA_VERSION = "mnist-conv-lr-study/v4"
LR_ARCHITECTURE_RELATIVE_STUDY_SCHEMA_VERSION = "mnist-conv-lr-study/v5"
LR_CONV2_TWO_RHO_STUDY_SCHEMA_VERSION = "mnist-conv-lr-study/v6"
LR_CONV3_SCHEME_TWO_RHO_STUDY_SCHEMA_VERSION = "mnist-conv-lr-study/v7"
LR_STUDY_ID_SCHEMA = "mnist-conv-lr-study-id/v1"

_V1_TOP_LEVEL_KEYS = {
    "schema_version",
    "name",
    "protocol_id",
    "status",
    "dataset",
    "model",
    "solver",
    "optimizer",
    "probe",
    "range_test",
    "candidate_training",
    "selection",
    "artifacts",
    "rows",
}

_V2_TOP_LEVEL_KEYS = _V1_TOP_LEVEL_KEYS | {"rescue"}

_V5_TOP_LEVEL_KEYS = _V1_TOP_LEVEL_KEYS | {
    "anchor_audit",
    "target_policies",
    "execution",
}

_V6_TOP_LEVEL_KEYS = _V1_TOP_LEVEL_KEYS | {
    "reuse",
    "conv1_settled",
    "rho_grid",
    "execution",
}

_V7_TOP_LEVEL_KEYS = _V1_TOP_LEVEL_KEYS | {
    "initialization",
    "rho_grid",
    "stages",
    "preflight",
    "execution",
}

# The readable canonical values live in
# configs/conv/hardsigmoid_lr_study_sgd_bs16_v1.json.  Hashes make every
# section strict without maintaining a second, subtly divergent copy of the
# long scientific contract in Python.  Rows are additionally exposed below
# and checked field by field because they are the primary frozen handoff.
_FROZEN_SECTION_SHA256 = {
    "status": "f0a9b82826a97cb6a39fc03062555154a80065c2b677a3839be900ba50111561",
    "dataset": "8282c94b0a4ad7b27e22cd0f483be691c40180e275b321b7dc8db5809687afc9",
    "model": "aee2038df879e566c6176bf63ed38f57582d768e35fcca1e53a505467825ac79",
    "solver": "ba7fe060f305aba78a1c4d505b92de85323d5954b0853fb269b742a0f8a748ab",
    "optimizer": "d85e92e869059b7d06f26d951df5c255bd8f64cf065970fc7a4b5eca53b217c3",
    "probe": "a9c1066d6663f78c70283d4976bad0153b6c022a25718cccd36d82f27b41c007",
    "range_test": "5d711cdcbb0290ab0cdcbf669f14d224a38c7382e2b0b374dd425893c1f27b63",
    "candidate_training": "56c1291c27cab5347fdc8c0041bbbd9ef751827793c28f323a8a0b1f2df9707a",
    "selection": "5cabc28bbcf348625eaee292e7373346e8a0f9daa71ef0e57ffc718a811c5b68",
    "artifacts": "99515d7cae57e1a53585181a81832d0a015186714ef39aaf99c87267d9562977",
}

# The v2 identity is deliberately a separate, strict rescue contract.  It
# changes only the unresolved-row range schedule and binds every reused input
# to the completed parent v1 study.  Keeping a second digest table makes v1
# validation and identity fully backward compatible.
_FROZEN_RESCUE_SECTION_SHA256 = {
    "status": "315593cb95898fa768efb6a8c0338527c84f535eb1746706bf12393798eeaba3",
    "dataset": "8282c94b0a4ad7b27e22cd0f483be691c40180e275b321b7dc8db5809687afc9",
    "model": "45f7d1ae9ff2b04c0f3fe313acf290f8884b6d0c2ba4a1579da200c881db79b1",
    "solver": "ba7fe060f305aba78a1c4d505b92de85323d5954b0853fb269b742a0f8a748ab",
    "optimizer": "d85e92e869059b7d06f26d951df5c255bd8f64cf065970fc7a4b5eca53b217c3",
    "probe": "a9c1066d6663f78c70283d4976bad0153b6c022a25718cccd36d82f27b41c007",
    "range_test": "818e34211b6e5473a72812caf64539f628698224cb8fe7d1371d94369ea189b6",
    "candidate_training": "56c1291c27cab5347fdc8c0041bbbd9ef751827793c28f323a8a0b1f2df9707a",
    "selection": "5cabc28bbcf348625eaee292e7373346e8a0f9daa71ef0e57ffc718a811c5b68",
    "artifacts": "0a5a505b81d99dc9f39f31a0f050b31b5741c3b3ce50f96776e7c0e8ee12cb99",
    "rescue": "a37ebeebad3998b428fd29c37ec8e8a4a0ba69acac749215c9f6ba055431de2d",
}

# Filled from the readable canonical v3 config.  V3 changes the LR coordinate
# rather than mutating either completed span-normalized study.
_FROZEN_RELATIVE_RHO_SECTION_SHA256 = {
    "status": "6f31ddeb13b7f45600f106afc9b8d4b13b9c173bee58bcd2cab6dc8b39709b9c",
    "dataset": "8282c94b0a4ad7b27e22cd0f483be691c40180e275b321b7dc8db5809687afc9",
    "model": "7ad491b30f580cf4c1955f9b93e9e2bc1003c68e58d422d22c324414d2c00ffe",
    "solver": "ba7fe060f305aba78a1c4d505b92de85323d5954b0853fb269b742a0f8a748ab",
    "optimizer": "d85e92e869059b7d06f26d951df5c255bd8f64cf065970fc7a4b5eca53b217c3",
    "probe": "3ba33cdf3a8f992cdbe111c4bf93d538d295061d6bb9bdffb8b65e2d02df766c",
    "range_test": "07a096915ee8ffb9a014e7c7cf3a2cd3c41ace81125cd9e9feb1e2823c2297a4",
    "candidate_training": "aead3720cce76ad5c73bf6469183fa5bd2a8ee093c05bbc362dd9401b9c45e1e",
    "selection": "5cabc28bbcf348625eaee292e7373346e8a0f9daa71ef0e57ffc718a811c5b68",
    "artifacts": "aadccd644019990b034374a75f2879324e481403704b160eaba48eeeda22a4da",
    "rescue": "9740857081096bff2df913714ca5c6231b362a24f29667cfda0e106c4d5fbf39",
}

_FROZEN_LAYERWISE_RHO_SECTION_SHA256 = {
    "status": "450f92c34d7c0f945ff38f5bf8ff81f689d0ebb19082a8f2e6da4bb727c99281",
    "dataset": "8282c94b0a4ad7b27e22cd0f483be691c40180e275b321b7dc8db5809687afc9",
    "model": "36322ee36e632f603f0d6b889ad492d896ac162c42bc2de5c3c6ae646f0d79a0",
    "solver": "ba7fe060f305aba78a1c4d505b92de85323d5954b0853fb269b742a0f8a748ab",
    "optimizer": "9b3ab66f069135976a2d1405c623ceb94a0aad795b99930db54b607958035e54",
    "probe": "0247be1965a6967225de1fff26fb5f79b761124f12ab684ca8a76188907de572",
    "range_test": "6547b7d06244bcb233883061b1a12f726940e567efcb4d06f4be707a0e832994",
    "candidate_training": "43d81302d07f84cf5dc61c4769fe2c4d367fd8c34e6c3f502b7d688f6832ce4a",
    "selection": "41c8164e48f72b1a7d4d37830e5252babccf45e0186e05890d1eebb4b96e4a2e",
    "artifacts": "fef437b1074a445cd52e23d1b36faf31f70e8c5250362c199e9b74d3aa762d2a",
}

# Populated from the readable canonical v5 config.  V5 is an ordinary-MNIST
# optimization diagnostic with an architecture-level alpha; it is deliberately
# content-addressed separately from the immutable v1--v4 contracts above.
_FROZEN_ARCHITECTURE_RELATIVE_SECTION_SHA256 = {
    "status": "60bd009ced6deee94042d2f221d67845218729501790c464cea45949e855a510",
    "dataset": "12c5c3a48ccffa3b0d2797ce0a628e29423037c585217340a5c1f03012fc43be",
    "model": "9f9a727acd6a39c6617b255b91f1b3c69726c71439c433879148a67b4389582c",
    "solver": "ba7fe060f305aba78a1c4d505b92de85323d5954b0853fb269b742a0f8a748ab",
    "optimizer": "440b998018db7d6ff5696b85c09a752b44ba17455834dcaf0a03f6450d081d28",
    "anchor_audit": "7c482d7fd75c6f8f493041680ec4ad833cad94389d6a2e1a67ab147257131cab",
    "probe": "4a1fe29e1c598c24f73b1f5b5058273400d0a7f5f551e63b45d5bd3ec810b537",
    "range_test": "564e258ee8c6878ea8753d9c58c55f002e93270289b5c921bc7beb9b07875865",
    "target_policies": "af35551e2eec1a2aaf5cd1a2b831bf0185e99681b9e5128953bbef0e950ddf48",
    "candidate_training": "f73c07b15c730534c788f63f4c339bd0f9cf35ec0709e0195502dc4ebc4e6235",
    "selection": "a15d20dd829d9d3071f4309937ce445282680c5264332c6e045e6fd5a61d8108",
    "execution": "49f0be22c87f123af716d660e20e9994518c8dc2ba903df725823d2f728172bb",
    "artifacts": "3d88c21f670146c82ba902c2ff89044783ff0bb8396a38d286721d6a5cef6119",
}

# V6 is a separate ordinary-MNIST Conv2 diagnostic.  It removes the v5
# architecture-alpha/profile coordinate and directly freezes independent
# convolutional and dense relative-update targets.  The parent v5 artifact
# hashes are scientific inputs, so the complete reuse policy is hashed too.
_FROZEN_CONV2_TWO_RHO_SECTION_SHA256 = {
    "status": "affc5f3be0902a0e44fa7427d63bf08c32219bae6a92cbd66dc1140377546a75",
    "dataset": "12c5c3a48ccffa3b0d2797ce0a628e29423037c585217340a5c1f03012fc43be",
    "model": "a72076ecc61f02e7408401cca24bd48031f7d67f74ba6cad8ab924f7c4857e5b",
    "solver": "ba7fe060f305aba78a1c4d505b92de85323d5954b0853fb269b742a0f8a748ab",
    "optimizer": "083c2be0fb29040090189005fefa0cb2c211aef8f842a04cc31121467eca879f",
    "reuse": "066549e26278c44b8906f10c4deb283b9149edaaa6b7fd34d5c7995281c440e9",
    "conv1_settled": "9785b521daf6b02308725e827feeb4929a5228f32e6bc471a56e4379fbd0b62f",
    "probe": "4445922b3576fab3ad15780ec3cc3888ebbdec51d19ac0834a5147da9e944647",
    "range_test": "fbeec1e8e8200aae6e526c443c4adb40eb6c47a7444a158f7a634917015e458b",
    "rho_grid": "3a69a2f13f6d0f20d88cf501c574ffdb8a222862f75febfcb7fc3522af38e416",
    "candidate_training": "74e6d308d0a4e99b605b22b7633b0da835846fa5c4fe49448a61aea55492a226",
    "selection": "cb73cb39bde1c6aa3e49dd9dbec449fdccddade2e35244680cf31073b071deed",
    "execution": "db0915e1b6da81a62b853b02641367c48fde380e3973e827df84e47e774ca3fb",
    "artifacts": "2a041ac9f3e1f08b4ccba2f72fa1f8f14d0077972fba9bf9d279e377ee94051e",
}

# V7 is a separately content-addressed ordinary-MNIST Conv3 study.  It
# independently selects direct convolutional and dense relative-update
# targets for all three schemes, with one bounded upper-grid expansion.
_FROZEN_CONV3_SCHEME_TWO_RHO_SECTION_SHA256 = {
    "status": "45a0152cacfad43ef110be04abf21a0aa54f0879f08c6d9a7d807d7fa722d428",
    "dataset": "335acf6c07329f3e54a1141f2faa3dfd980947fce39a44109f44a9b2bd30e50a",
    "model": "c38d6d4719fc608736ca8702c22db776bdab3ce5120d740873084677a798fd95",
    "solver": "ba7fe060f305aba78a1c4d505b92de85323d5954b0853fb269b742a0f8a748ab",
    "optimizer": "25d2f346a9660d41b14d1b1bc22d9ccb38987e0ea540d1ecb77e237a6fbd7fb8",
    "initialization": "e541c5390ab60f3cce76026daec5f474a90396f7ff1f2876b9c6f4c1e69e94eb",
    "probe": "60d944540ccec9f936da22c6f39eb32e5dfb98f69d26a8b3123ea70e5081f3cc",
    "range_test": "fbeec1e8e8200aae6e526c443c4adb40eb6c47a7444a158f7a634917015e458b",
    "rho_grid": "284ac55ebb53c0451cd4fbe942d38ec3cd1f15056a392891a3dd01d0e266136b",
    "candidate_training": "f5d59da54c1ee97df91440eb0089df31395beb9392d08e6d6175cc6e706f9b67",
    "selection": "ce0eb86fdf375e22f7346058a199f041150bd6d8170ae01b4662e7540a1a2c17",
    "stages": "027f786e2cec88bd7c8b9fd52107465609fef21be32ae5410cd31360f6a9e7d0",
    "preflight": "db16785769ee82a4e515f85ae1dec88238849bc46e5922b7b7d79e16b5a04b10",
    "execution": "17b08bc92765968674019f459b4b735943c38d24ec057b261b4899835f94cc43",
    "artifacts": "61c88a25136cb6b4a2663abf2c61430ea3fa8ad7948ee908b0206a1f9474f12f",
}

FROZEN_ROWS: tuple[dict[str, Any], ...] = (
    {
        "row_id": "conv1_baseline_v1_c1",
        "architecture": "conv1",
        "scheme": "baseline",
        "run_name": "mnist_bp_amp_v1_c1",
        "voltage_amp": 1.0,
        "current_amp": 1.0,
        "input_gain": 75.6030807495,
        "inference_iterations": 4,
        "training_iterations": 4,
    },
    {
        "row_id": "conv1_ours_v4_c1",
        "architecture": "conv1",
        "scheme": "ours",
        "run_name": "mnist_bp_amp_v4_c1",
        "voltage_amp": 4.0,
        "current_amp": 1.0,
        "input_gain": 84.8402175903,
        "inference_iterations": 4,
        "training_iterations": 4,
    },
    {
        "row_id": "conv1_legacy_v4_c0p25",
        "architecture": "conv1",
        "scheme": "legacy",
        "run_name": "mnist_bp_amp_v4_c0p25",
        "voltage_amp": 4.0,
        "current_amp": 0.25,
        "input_gain": 31.8188591003,
        "inference_iterations": 4,
        "training_iterations": 4,
    },
    {
        "row_id": "conv2_baseline_v1_c1",
        "architecture": "conv2",
        "scheme": "baseline",
        "run_name": "mnist_bp_amp_v1_c1",
        "voltage_amp": 1.0,
        "current_amp": 1.0,
        "input_gain": 253.302230835,
        "inference_iterations": 16,
        "training_iterations": 6,
    },
    {
        "row_id": "conv2_ours_v4_c1",
        "architecture": "conv2",
        "scheme": "ours",
        "run_name": "mnist_bp_amp_v4_c1",
        "voltage_amp": 4.0,
        "current_amp": 1.0,
        "input_gain": 716.3439331055,
        "inference_iterations": 24,
        "training_iterations": 6,
    },
    {
        "row_id": "conv2_legacy_v4_c0p25",
        "architecture": "conv2",
        "scheme": "legacy",
        "run_name": "mnist_bp_amp_v4_c0p25",
        "voltage_amp": 4.0,
        "current_amp": 0.25,
        "input_gain": 661.4369506836,
        "inference_iterations": 8,
        "training_iterations": 4,
    },
)

FROZEN_RESCUE_ROWS: tuple[dict[str, Any], ...] = tuple(
    row
    for row in FROZEN_ROWS
    if row["row_id"] in {"conv1_ours_v4_c1", "conv1_legacy_v4_c0p25"}
)

FROZEN_RELATIVE_RHO_ROWS: tuple[dict[str, Any], ...] = tuple(
    row
    for row in FROZEN_ROWS
    if row["row_id"]
    in {
        "conv1_ours_v4_c1",
        "conv1_legacy_v4_c0p25",
        "conv2_ours_v4_c1",
        "conv2_legacy_v4_c0p25",
    }
)

FROZEN_LAYERWISE_RHO_ROWS: tuple[dict[str, Any], ...] = FROZEN_ROWS
FROZEN_ARCHITECTURE_RELATIVE_ROWS: tuple[dict[str, Any], ...] = FROZEN_ROWS
FROZEN_CONV2_TWO_RHO_ROWS: tuple[dict[str, Any], ...] = tuple(
    row for row in FROZEN_ROWS if row["architecture"] == "conv2"
)

FROZEN_CONV3_SCHEME_TWO_RHO_ROWS: tuple[dict[str, Any], ...] = (
    {
        "row_id": "conv3_baseline_v1_c1",
        "architecture": "conv3",
        "scheme": "baseline",
        "run_name": "mnist_bp_amp_v1_c1",
        "voltage_amp": 1.0,
        "current_amp": 1.0,
        "input_gain": 251.306137085,
        "inference_iterations": 24,
        "training_iterations": 8,
    },
    {
        "row_id": "conv3_ours_v4_c1",
        "architecture": "conv3",
        "scheme": "ours",
        "run_name": "mnist_bp_amp_v4_c1",
        "voltage_amp": 4.0,
        "current_amp": 1.0,
        "input_gain": 744.739074707,
        "inference_iterations": 32,
        "training_iterations": 8,
    },
    {
        "row_id": "conv3_legacy_v4_c0p25",
        "architecture": "conv3",
        "scheme": "legacy",
        "run_name": "mnist_bp_amp_v4_c0p25",
        "voltage_amp": 4.0,
        "current_amp": 0.25,
        "input_gain": 665.0302124023,
        "inference_iterations": 8,
        "training_iterations": 6,
    },
)


def _error(expected: str, provided: Any, path: str) -> SpecValidationError:
    return SpecValidationError(
        f"Expected {path} to be {expected}. Provided value: {provided!r}."
    )


def _strict_json_loads(text: str, source: Path) -> Any:
    def reject_constant(value: str) -> Any:
        raise _error("a finite JSON number", value, str(source))

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _error("JSON objects with unique keys", key, str(source))
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except json.JSONDecodeError as exc:
        raise SpecValidationError(
            f"Expected {source} to contain valid strict JSON. Provided error: {exc}."
        ) from exc


def _reject_non_finite(value: Any, path: str = "study") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise _error("a finite JSON value", value, path)
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")


def _section_sha256(value: Any) -> str:
    import hashlib

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _assert_exact(expected: Any, provided: Any, path: str) -> None:
    """Recursively compare values without Python's bool/int equivalence."""

    if type(expected) is not type(provided):
        raise _error(f"exactly {expected!r}", provided, path)
    if isinstance(expected, dict):
        if set(expected) != set(provided):
            missing = sorted(set(expected) - set(provided))
            extra = sorted(set(provided) - set(expected))
            raise _error(
                f"an object with exactly keys {sorted(expected)!r}",
                {"missing": missing, "extra": extra},
                path,
            )
        for key in expected:
            _assert_exact(expected[key], provided[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(provided):
            raise _error(f"a sequence of length {len(expected)}", provided, path)
        for index, (expected_item, provided_item) in enumerate(zip(expected, provided)):
            _assert_exact(expected_item, provided_item, f"{path}[{index}]")
        return
    if expected != provided:
        raise _error(f"exactly {expected!r}", provided, path)


@dataclass(frozen=True)
class LRStudySpec:
    """Validated, immutable-by-copy representation of an LR study config."""

    data: dict[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LRStudySpec":
        if not isinstance(value, Mapping):
            raise _error("a JSON object", value, "study")
        study = copy.deepcopy(dict(value))
        _reject_non_finite(study)
        schema_version = study.get("schema_version")
        if schema_version == LR_STUDY_SCHEMA_VERSION:
            top_level_keys = _V1_TOP_LEVEL_KEYS
            protocol_id = "conv-hardsigmoid-lr-sgd-bs16-v1"
            section_digests = _FROZEN_SECTION_SHA256
            frozen_rows = FROZEN_ROWS
        elif schema_version == LR_RESCUE_STUDY_SCHEMA_VERSION:
            top_level_keys = _V2_TOP_LEVEL_KEYS
            protocol_id = "conv-hardsigmoid-lr-sgd-bs16-warm-in-rescue-v2"
            section_digests = _FROZEN_RESCUE_SECTION_SHA256
            frozen_rows = FROZEN_RESCUE_ROWS
        elif schema_version == LR_RELATIVE_RHO_STUDY_SCHEMA_VERSION:
            top_level_keys = _V2_TOP_LEVEL_KEYS
            protocol_id = "conv-hardsigmoid-lr-sgd-bs16-relative-rho-v3"
            section_digests = _FROZEN_RELATIVE_RHO_SECTION_SHA256
            frozen_rows = FROZEN_RELATIVE_RHO_ROWS
        elif schema_version == LR_LAYERWISE_RHO_STUDY_SCHEMA_VERSION:
            top_level_keys = _V1_TOP_LEVEL_KEYS
            protocol_id = (
                "conv-hardsigmoid-lr-layerwise-relative-rho-constant-sgd-bs16-v4"
            )
            section_digests = _FROZEN_LAYERWISE_RHO_SECTION_SHA256
            frozen_rows = FROZEN_LAYERWISE_RHO_ROWS
        elif schema_version == LR_ARCHITECTURE_RELATIVE_STUDY_SCHEMA_VERSION:
            top_level_keys = _V5_TOP_LEVEL_KEYS
            protocol_id = (
                "conv-hardsigmoid-lr-architecture-relative-median-constant-"
                "sgd-bs16-v5"
            )
            section_digests = _FROZEN_ARCHITECTURE_RELATIVE_SECTION_SHA256
            frozen_rows = FROZEN_ARCHITECTURE_RELATIVE_ROWS
        elif schema_version == LR_CONV2_TWO_RHO_STUDY_SCHEMA_VERSION:
            top_level_keys = _V6_TOP_LEVEL_KEYS
            protocol_id = (
                "conv-hardsigmoid-lr-conv2-two-rho-median-constant-"
                "sgd-bs16-v6"
            )
            section_digests = _FROZEN_CONV2_TWO_RHO_SECTION_SHA256
            frozen_rows = FROZEN_CONV2_TWO_RHO_ROWS
        elif schema_version == LR_CONV3_SCHEME_TWO_RHO_STUDY_SCHEMA_VERSION:
            top_level_keys = _V7_TOP_LEVEL_KEYS
            protocol_id = (
                "conv-hardsigmoid-lr-conv3-scheme-two-rho-median-constant-"
                "sgd-bs16-v7"
            )
            section_digests = _FROZEN_CONV3_SCHEME_TWO_RHO_SECTION_SHA256
            frozen_rows = FROZEN_CONV3_SCHEME_TWO_RHO_ROWS
        else:
            raise _error(
                (
                    f"one of {LR_STUDY_SCHEMA_VERSION!r}, "
                    f"{LR_RESCUE_STUDY_SCHEMA_VERSION!r}, "
                    f"{LR_RELATIVE_RHO_STUDY_SCHEMA_VERSION!r}, "
                    f"{LR_LAYERWISE_RHO_STUDY_SCHEMA_VERSION!r}, "
                    f"{LR_ARCHITECTURE_RELATIVE_STUDY_SCHEMA_VERSION!r}, or "
                    f"{LR_CONV2_TWO_RHO_STUDY_SCHEMA_VERSION!r}, or "
                    f"{LR_CONV3_SCHEME_TWO_RHO_STUDY_SCHEMA_VERSION!r}"
                ),
                schema_version,
                "study.schema_version",
            )

        missing = sorted(top_level_keys - set(study))
        extra = sorted(set(study) - top_level_keys)
        if missing or extra:
            raise _error(
                f"an object with exactly keys {sorted(top_level_keys)!r}",
                {"missing": missing, "extra": extra},
                "study",
            )
        name = study["name"]
        if not isinstance(name, str) or not name.strip():
            raise _error("a non-empty string", name, "study.name")
        study["name"] = name.strip()
        if study["protocol_id"] != protocol_id:
            raise _error(
                f"exactly {protocol_id!r}",
                study["protocol_id"],
                "study.protocol_id",
            )

        for section, expected_digest in section_digests.items():
            provided_digest = _section_sha256(study[section])
            if provided_digest != expected_digest:
                raise _error(
                    f"the frozen {section} contract with SHA-256 {expected_digest}",
                    {"sha256": provided_digest, "value": study[section]},
                    f"study.{section}",
                )

        _assert_exact(list(frozen_rows), study["rows"], "study.rows")
        return cls(study)

    @classmethod
    def from_file(cls, path: str | Path) -> "LRStudySpec":
        source = Path(path)
        return cls.from_dict(_strict_json_loads(source.read_text(), source))

    # Match the path-oriented naming used by staged orchestration callers.
    from_path = from_file

    def identity_payload(self) -> dict[str, Any]:
        """Return scientific content, excluding only the display name."""

        payload = copy.deepcopy(self.data)
        payload.pop("name")
        return payload

    @property
    def study_id(self) -> str:
        return study_fingerprint(self)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.data["rows"])


# Short alias for callers that use the generic naming convention of RunSpec.
StudySpec = LRStudySpec


def study_fingerprint(spec: LRStudySpec) -> str:
    """Return the display-name-independent content address for a study."""

    payload = {
        "schema_version": LR_STUDY_ID_SCHEMA,
        "study_spec": spec.identity_payload(),
    }
    return "lrstudy_" + sha256_json(payload)
