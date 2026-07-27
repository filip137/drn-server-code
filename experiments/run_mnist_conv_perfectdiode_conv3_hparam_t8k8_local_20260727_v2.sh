#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/filip/server_code_conv_learning_rate_protocol
source_root=/tmp/pd_conv3_lr_t8k8_20260727_source
python_bin=/home/filip/miniconda3/envs/py312/bin/python
controller=${source_root}/experiments/run_mnist_conv_perfectdiode_conv3_hparam_v2.py
worker=${source_root}/experiments/run_mnist_conv_perfectdiode_hparam_v2_worker.py
orchestrator=${source_root}/experiments/run_mnist_conv_perfectdiode_hparam_v2_orchestration.py

experiment_id=perfectdiode-conv3-lr-ordinary-mnist-t8k8-20260727-v3
study_id=lrstudy_294f233afd76a4075482d5e6d43c36f74739e63b88ecdf5cc8b6e3091fa323f6
study_name=perfect-diode-conv3-ordinary-mnist-sgd-adam-high-rho-screen
bundle_root=${repo_root}/results/perfectdiode_conv3_lr_t8k8_20260727_v3
study_root=${bundle_root}/perfectdiode_hparam_studies/${study_name}--${study_id}
study=${study_root}/study.resolved.json
surface_manifest=${study_root}/surface_manifest.json
shared_asset_hashes=${study_root}/shared_asset_hashes.json
assets_dir=${study_root}/upstream_tk/shared_assets
environment_contract=${study_root}/environment/contract.json
main_environment=${study_root}/environment/main.json
akib_environment=${study_root}/environment/akibscomputer.json
source_archive=/tmp/pd_conv3_lr_t8k8_20260727_source.tar.gz
data_root=/home/filip/server_code/data
plan=${repo_root}/docs/experiment_plans/perfectdiode-conv3-lr-ordinary-mnist-t8k8-20260727-v3.md
tracker=${repo_root}/docs/current_experiments.md
plan_validator=${repo_root}/skills/run-experiment-pipeline/scripts/validate_experiment_plan.py
tracker_validator=${repo_root}/skills/run-experiment-pipeline/scripts/validate_current_experiments.py
runner_path=${repo_root}/experiments/run_mnist_conv_perfectdiode_conv3_hparam_t8k8_local_20260727_v2.sh
scheduled_receipt=${study_root}/preflight/scheduled-run-preflight.json

main_baseline_gate=${study_root}/preflight/fixed_tk_gate/main/conv3_baseline_v1_c1/completion.json
main_legacy_gate=${study_root}/preflight/fixed_tk_gate/main/conv3_legacy_v4_c0p25/completion.json
main_smoke=${study_root}/preflight/representative_canary/main/completion.json

remote_host=akib
remote_source_root=/home/filiposana/staged/pd_conv3_lr_t8k8_20260727_source
remote_python=/home/filiposana/miniconda3/envs/py312/bin/python
remote_controller=${remote_source_root}/experiments/run_mnist_conv_perfectdiode_conv3_hparam_v2.py
remote_worker=${remote_source_root}/experiments/run_mnist_conv_perfectdiode_hparam_v2_worker.py
remote_orchestrator=${remote_source_root}/experiments/run_mnist_conv_perfectdiode_hparam_v2_orchestration.py
remote_bundle_root=/home/filiposana/results/perfectdiode_conv3_lr_t8k8_20260727_v3
remote_study_root=${remote_bundle_root}/perfectdiode_hparam_studies/${study_name}--${study_id}
remote_study=${remote_study_root}/study.resolved.json
remote_surface_manifest=${remote_study_root}/surface_manifest.json
remote_assets_dir=${remote_study_root}/upstream_tk/shared_assets
remote_environment_contract=${remote_study_root}/environment/contract.json
remote_akib_environment=${remote_study_root}/environment/akibscomputer.json
remote_source_archive=/home/filiposana/staged/pd_conv3_lr_t8k8_20260727_source.tar.gz
remote_data_root=/home/filiposana/server_code/data
remote_ours_gate=${remote_study_root}/preflight/fixed_tk_gate/akibscomputer/conv3_ours_v4_c1/completion.json
remote_smoke=${remote_study_root}/preflight/representative_canary/akibscomputer/completion.json
delegated_pane_token=%akibscomputer-0.0-v1

stages=(
  audit
  import_tk
  optimizer_probe
  rho_canary_core
  rho_core_candidates
  select_core
  rho_canary_expansion
  rho_expansion_candidates
  select_expanded
  post_training_tk
  finalize_lr
)
main_surfaces=(
  conv3_baseline_v1_c1--sgd
  conv3_baseline_v1_c1--adam
  conv3_legacy_v4_c0p25--sgd
  conv3_legacy_v4_c0p25--adam
)
akib_surfaces=(
  conv3_ours_v4_c1--sgd
  conv3_ours_v4_c1--adam
)

local_worker_common=(
  env
  KMP_DISABLE_SHM=1
  KMP_SHM_DISABLE=1
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  PYTHONUNBUFFERED=1
  "${python_bin}"
  "${worker}"
  --study "${study}"
  --surface-manifest "${surface_manifest}"
  --host main
  --source-archive "${source_archive}"
  --environment-contract "${environment_contract}"
  --execution-environment "${main_environment}"
  --assets-dir "${assets_dir}"
  --data-root "${data_root}"
  --device cuda
)
remote_worker_common=(
  ssh
  -F /home/filip/.ssh/config
  -o BatchMode=yes
  "${remote_host}"
  env
  PD_LR_TMUX_SESSION=akibscomputer
  PD_LR_TMUX_PANE_TOKEN="${delegated_pane_token}"
  KMP_DISABLE_SHM=1
  KMP_SHM_DISABLE=1
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  PYTHONUNBUFFERED=1
  "${remote_python}"
  "${remote_worker}"
  --study "${remote_study}"
  --surface-manifest "${remote_surface_manifest}"
  --host akibscomputer
  --source-archive "${remote_source_archive}"
  --environment-contract "${remote_environment_contract}"
  --execution-environment "${remote_akib_environment}"
  --assets-dir "${remote_assets_dir}"
  --data-root "${remote_data_root}"
  --device cuda
)

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

manifest_value() {
  "${python_bin}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["execution_source"][sys.argv[2]])' \
    "${surface_manifest}" "$1"
}

require_sha256() {
  local path=$1
  local expected=$2
  local observed
  observed=$(sha256_file "${path}")
  if [[ "${observed}" != "${expected}" ]]; then
    printf 'Expected SHA-256 %s for %s. Provided value: %s.\n' \
      "${expected}" "${path}" "${observed}" >&2
    return 1
  fi
}

require_idle_pane() {
  local pane=$1
  local command
  command=$(tmux display-message -p -t "${pane}" '#{pane_current_command}')
  if [[ "${command}" != "bash" ]]; then
    printf 'Expected tmux pane %s to be idle with command bash. Provided value: %s.\n' \
      "${pane}" "${command}" >&2
    return 1
  fi
}

verify_scheduled_receipt() {
  local receipt_runner_sha
  local current_runner_sha
  test -f "${scheduled_receipt}"
  receipt_runner_sha=$("${python_bin}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["runner_sha256"])' \
    "${scheduled_receipt}")
  current_runner_sha=$(sha256_file "${runner_path}")
  if [[ "${receipt_runner_sha}" != "${current_runner_sha}" ]]; then
    printf 'Expected runner SHA-256 %s from the scheduled receipt. Provided value: %s.\n' \
      "${receipt_runner_sha}" "${current_runner_sha}" >&2
    return 1
  fi
}

require_tracker_launch_ready() {
  "${python_bin}" "${tracker_validator}" "${tracker}" \
    --require-experiment-id "${experiment_id}" \
    --require-launch-ready
}

tk_gate_main() {
  "${local_worker_common[@]}" \
    --surface-id conv3_baseline_v1_c1--adam \
    --preflight-tk-gate
  "${local_worker_common[@]}" \
    --surface-id conv3_legacy_v4_c0p25--adam \
    --preflight-tk-gate
}

tk_gate_akibscomputer() {
  "${remote_worker_common[@]}" \
    --surface-id conv3_ours_v4_c1--adam \
    --preflight-tk-gate
}

require_fixed_tk_main_receipts() {
  test -f "${main_baseline_gate}"
  test -f "${main_legacy_gate}"
}

require_fixed_tk_akibscomputer_receipt() {
  ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    test -f "${remote_ours_gate}"
}

validate_fixed_tk_main() {
  require_fixed_tk_main_receipts
  tk_gate_main >/dev/null
}

validate_fixed_tk_akibscomputer() {
  require_fixed_tk_akibscomputer_receipt
  tk_gate_akibscomputer >/dev/null
}

smoke_main() {
  "${local_worker_common[@]}" \
    --surface-id conv3_baseline_v1_c1--adam \
    --preflight-canary
}

smoke_akibscomputer() {
  "${remote_worker_common[@]}" \
    --surface-id conv3_ours_v4_c1--adam \
    --preflight-canary
}

require_smoke_receipts() {
  test -f "${main_smoke}"
  ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    test -f "${remote_smoke}"
}

validate_smoke_receipts() {
  require_smoke_receipts
  smoke_main >/dev/null
  smoke_akibscomputer >/dev/null
}

preflight() {
  local source_commit
  local source_archive_sha256
  local manifest_sha256
  local study_sha256
  local environment_contract_sha256
  local main_environment_sha256
  local akib_environment_sha256

  test -x "${python_bin}"
  test -f "${controller}"
  test -f "${worker}"
  test -f "${orchestrator}"
  test -f "${study}"
  test -f "${surface_manifest}"
  test -f "${shared_asset_hashes}"
  test -f "${environment_contract}"
  test -f "${main_environment}"
  test -f "${akib_environment}"
  test -f "${source_archive}"
  test -d "${assets_dir}"
  test -d "${data_root}/MNIST/raw"
  test -f "${plan}"
  test -f "${tracker}"
  test -f "${plan_validator}"
  test -f "${tracker_validator}"
  test -f "${runner_path}"

  source_commit=$(manifest_value source_commit)
  source_archive_sha256=$(manifest_value source_archive_sha256)
  study_sha256=$(manifest_value resolved_study_sha256)
  environment_contract_sha256=$(manifest_value environment_contract_sha256)
  main_environment_sha256=$("${python_bin}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["execution_source"]["host_environment_sha256s"]["main"])' \
    "${surface_manifest}")
  akib_environment_sha256=$("${python_bin}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["execution_source"]["host_environment_sha256s"]["akibscomputer"])' \
    "${surface_manifest}")
  manifest_sha256=$(sha256_file "${surface_manifest}")

  test "$(git -C "${source_root}" rev-parse HEAD)" = "${source_commit}"
  test -z "$(git -C "${source_root}" status --porcelain)"
  require_sha256 "${source_archive}" "${source_archive_sha256}"
  require_sha256 "${study}" "${study_sha256}"
  require_sha256 "${environment_contract}" "${environment_contract_sha256}"
  require_sha256 "${main_environment}" "${main_environment_sha256}"
  require_sha256 "${akib_environment}" "${akib_environment_sha256}"

  "${python_bin}" "${controller}" validate --study "${study}" >/dev/null
  "${python_bin}" "${controller}" preflight-plan \
    --manifest "${surface_manifest}" --host main >/dev/null
  "${python_bin}" "${controller}" preflight-plan \
    --manifest "${surface_manifest}" --host akibscomputer >/dev/null
  "${python_bin}" "${orchestrator}" plan --root "${study_root}" >/dev/null
  "${python_bin}" "${orchestrator}" status \
    --root "${study_root}" --host main >/dev/null
  "${python_bin}" "${plan_validator}" \
    "${plan}" --require-approved --verify-files
  "${python_bin}" "${tracker_validator}" "${tracker}" >/dev/null
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader >/dev/null
  require_idle_pane main:0.0
  require_idle_pane akibscomputer:0.0

  ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    test -x "${remote_python}"
  ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    test -f "${remote_controller}"
  ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    test -f "${remote_worker}"
  ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    test -f "${remote_orchestrator}"
  ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    test -f "${remote_study}"
  ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    test -f "${remote_surface_manifest}"
  ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    test -d "${remote_assets_dir}"
  ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    test -d "${remote_data_root}/MNIST/raw"
  test "$(ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    git -C "${remote_source_root}" rev-parse HEAD)" = "${source_commit}"
  test -z "$(ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    git -C "${remote_source_root}" status --porcelain)"
  test "$(ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    sha256sum "${remote_source_archive}" | awk '{print $1}')" = \
    "${source_archive_sha256}"
  test "$(ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    sha256sum "${remote_study}" | awk '{print $1}')" = "${study_sha256}"
  test "$(ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    sha256sum "${remote_surface_manifest}" | awk '{print $1}')" = \
    "${manifest_sha256}"
  test "$(ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    sha256sum "${remote_environment_contract}" | awk '{print $1}')" = \
    "${environment_contract_sha256}"
  test "$(ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    sha256sum "${remote_akib_environment}" | awk '{print $1}')" = \
    "${akib_environment_sha256}"
  ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    "${remote_python}" "${remote_orchestrator}" status \
    --root "${remote_study_root}" --host akibscomputer >/dev/null
  ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
    >/dev/null

  validate_fixed_tk_main
  validate_fixed_tk_akibscomputer
  validate_smoke_receipts

  printf 'Conv3 perfect-diode T=8/K=8 LR runner preflight passed.\n'
}

run_main_lane() {
  local surface
  local stage
  validate_fixed_tk_main
  require_tracker_launch_ready
  mkdir -p "${study_root}/logs"
  for surface in "${main_surfaces[@]}"; do
    for stage in "${stages[@]}"; do
      "${local_worker_common[@]}" \
        --surface-id "${surface}" --stage "${stage}" \
      2>&1 | tee -a "${study_root}/logs/main.log"
    done
  done
  "${python_bin}" "${orchestrator}" finalize-host \
    --root "${study_root}" --host main
}

run_akibscomputer_lane() {
  local surface
  local stage
  validate_fixed_tk_akibscomputer
  require_tracker_launch_ready
  mkdir -p "${study_root}/logs"
  for surface in "${akib_surfaces[@]}"; do
    for stage in "${stages[@]}"; do
      "${remote_worker_common[@]}" \
        --surface-id "${surface}" --stage "${stage}" \
      2>&1 | tee -a "${study_root}/logs/akibscomputer.log"
    done
  done
  ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    "${remote_python}" "${remote_orchestrator}" finalize-host \
    --root "${remote_study_root}" --host akibscomputer
}

status() {
  "${python_bin}" "${orchestrator}" status \
    --root "${study_root}" --host main
  ssh -F /home/filip/.ssh/config -o BatchMode=yes "${remote_host}" \
    "${remote_python}" "${remote_orchestrator}" status \
    --root "${remote_study_root}" --host akibscomputer
}

dispatch() {
  local current_runner_sha
  preflight
  verify_scheduled_receipt
  validate_fixed_tk_main
  validate_fixed_tk_akibscomputer
  validate_smoke_receipts
  require_idle_pane main:0.0
  require_idle_pane akibscomputer:0.0
  require_tracker_launch_ready
  current_runner_sha=$(sha256_file "${runner_path}")
  tmux send-keys -t main:0.0 "${runner_path} --lane main" C-m
  tmux send-keys -t akibscomputer:0.0 \
    "${runner_path} --lane akibscomputer" C-m
  printf 'Dispatched Conv3 T=8/K=8 LR runner %s to main:0.0 and akibscomputer:0.0.\n' \
    "${current_runner_sha}"
}

case "${1:-}" in
  --preflight)
    preflight
    ;;
  --tk-gate)
    case "${2:-}" in
      main) tk_gate_main ;;
      akibscomputer) tk_gate_akibscomputer ;;
      *)
        printf 'Expected --tk-gate host to be main or akibscomputer. Provided value: %s.\n' \
          "${2:-}" >&2
        exit 2
        ;;
    esac
    ;;
  --smoke)
    case "${2:-}" in
      main) smoke_main ;;
      akibscomputer) smoke_akibscomputer ;;
      *)
        printf 'Expected --smoke host to be main or akibscomputer. Provided value: %s.\n' \
          "${2:-}" >&2
        exit 2
        ;;
    esac
    ;;
  --lane)
    case "${2:-}" in
      main) run_main_lane ;;
      akibscomputer) run_akibscomputer_lane ;;
      *)
        printf 'Expected --lane host to be main or akibscomputer. Provided value: %s.\n' \
          "${2:-}" >&2
        exit 2
        ;;
    esac
    ;;
  --dispatch)
    dispatch
    ;;
  --status)
    status
    ;;
  *)
    printf 'Expected one of --preflight, --tk-gate HOST, --smoke HOST, --lane HOST, --dispatch, or --status. Provided value: %s.\n' \
      "${1:-}" >&2
    exit 2
    ;;
esac
