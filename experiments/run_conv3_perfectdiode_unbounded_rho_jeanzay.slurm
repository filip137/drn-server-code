#!/bin/bash -l
#SBATCH --account=fmu@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-dev
#SBATCH --constraint=v100-16g
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00

set -euo pipefail

source_root=${PD_RHO_SOURCE_ROOT:?PD_RHO_SOURCE_ROOT is required}
result_root=${PD_RHO_RESULT_ROOT:?PD_RHO_RESULT_ROOT is required}
mode=${PD_RHO_MODE:?PD_RHO_MODE is required}
task_id=${SLURM_ARRAY_TASK_ID:-${PD_RHO_TASK_ID:-}}

if [[ -z "${task_id}" ]]; then
  echo "Expected SLURM_ARRAY_TASK_ID or PD_RHO_TASK_ID." >&2
  exit 2
fi
if [[ ! "${task_id}" =~ ^[0-9]+$ ]]; then
  echo "Expected a non-negative integer task ID, got ${task_id}." >&2
  exit 2
fi

module purge
module load pytorch-gpu/py3/2.5.0

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "${source_root}"

surface_from_index() {
  local surface_index=$1
  case "${surface_index}" in
    0)
      run_name=mnist_bp_amp_v1_c1
      scheme=baseline
      optimizer=SGD
      rho_conv=(0.009 0.027 0.081)
      rho_dense=(0.03 0.09 0.27)
      ;;
    1)
      run_name=mnist_bp_amp_v1_c1
      scheme=baseline
      optimizer=Adam
      rho_conv=(0.009 0.027 0.081)
      rho_dense=(0.03 0.09 0.27)
      ;;
    2)
      run_name=mnist_bp_amp_v4_c1
      scheme=ours
      optimizer=SGD
      rho_conv=(0.009 0.027 0.081)
      rho_dense=(0.03 0.09 0.27)
      ;;
    3)
      run_name=mnist_bp_amp_v4_c1
      scheme=ours
      optimizer=Adam
      rho_conv=(0.009 0.027 0.081)
      rho_dense=(0.03 0.09 0.27)
      ;;
    4)
      run_name=mnist_bp_amp_v4_c0p25
      scheme=legacy
      optimizer=SGD
      rho_conv=(0.001 0.003 0.009)
      rho_dense=(0.0033333333333333335 0.01 0.03)
      ;;
    5)
      run_name=mnist_bp_amp_v4_c0p25
      scheme=legacy
      optimizer=Adam
      rho_conv=(0.001 0.003 0.009)
      rho_dense=(0.0033333333333333335 0.01 0.03)
      ;;
    *)
      echo "Expected surface index in [0, 5], got ${surface_index}." >&2
      exit 2
      ;;
  esac
  optimizer_token=${optimizer,,}
  surface_id="conv3_${scheme}--${optimizer_token}"
  config_path="${source_root}/run_inputs/perfect_diode/${run_name}/seed_0/source_config.json"
  surface_root="${result_root}/surfaces/${surface_id}"
}

apply_grid_overrides() {
  if [[ -n "${PD_RHO_CONV_VALUES:-}" ]]; then
    read -r -a rho_conv <<< "${PD_RHO_CONV_VALUES}"
  fi
  if [[ -n "${PD_RHO_DENSE_VALUES:-}" ]]; then
    read -r -a rho_dense <<< "${PD_RHO_DENSE_VALUES}"
  fi
  if [[ "${#rho_conv[@]}" -eq 0 || "${#rho_dense[@]}" -eq 0 ]]; then
    echo "Rho grids must each contain at least one value." >&2
    exit 2
  fi
}

select_one_surface() {
  local surface_index=${PD_RHO_SURFACE_INDEX:?PD_RHO_SURFACE_INDEX is required}
  surface_from_index "${surface_index}"
  apply_grid_overrides
}

run_surface() {
  local operation=$1
  shift
  python -m experiments.rho_search \
    "${config_path}" \
    --output-root "${surface_root}" \
    --rho-conv "${rho_conv[@]}" \
    --rho-dense "${rho_dense[@]}" \
    --optimizer "${optimizer}" \
    --bias-policy q90_cap \
    --probe-batches 32 64 128 \
    --stability-tolerance 0.10 \
    --epochs 3 \
    --validation-batch-size 64 \
    --split-seed 0 \
    --shuffle-seed 0 \
    --device cuda \
    "${operation}" "$@"
}

case "${mode}" in
  canary)
    if [[ "${task_id}" -ne 0 ]]; then
      echo "Canary mode requires task 0, got ${task_id}." >&2
      exit 2
    fi
    surface_from_index 0
    python -m experiments.rho_search \
      "${config_path}" \
      --output-root "${result_root}/canary" \
      --rho-conv 0.009 \
      --rho-dense 0.03 \
      --optimizer SGD \
      --bias-policy q90_cap \
      --probe-batches 2 \
      --stability-tolerance 1000000 \
      --epochs 1 \
      --max-batches 1 \
      --max-validation-batches 1 \
      --validation-batch-size 64 \
      --split-seed 0 \
      --shuffle-seed 0 \
      --device cuda
    ;;
  probe)
    surface_from_index "${task_id}"
    run_surface --probe-only
    ;;
  probe-one)
    if [[ "${task_id}" -ne 0 ]]; then
      echo "probe-one mode requires task 0, got ${task_id}." >&2
      exit 2
    fi
    select_one_surface
    run_surface --probe-only
    ;;
  cell)
    if [[ "${task_id}" -gt 53 ]]; then
      echo "Cell mode requires task ID in [0, 53], got ${task_id}." >&2
      exit 2
    fi
    surface_index=$((task_id / 9))
    cell_index=$((task_id % 9))
    surface_from_index "${surface_index}"
    run_surface --index "${cell_index}"
    ;;
  cell-one)
    select_one_surface
    cell_count=$(("${#rho_conv[@]}" * "${#rho_dense[@]}"))
    if [[ "${task_id}" -ge "${cell_count}" ]]; then
      echo "cell-one task ${task_id} is outside this ${cell_count}-cell grid." >&2
      exit 2
    fi
    run_surface --index "${task_id}"
    ;;
  collect)
    surface_from_index "${task_id}"
    run_surface --collect-only
    ;;
  collect-one)
    if [[ "${task_id}" -ne 0 ]]; then
      echo "collect-one mode requires task 0, got ${task_id}." >&2
      exit 2
    fi
    select_one_surface
    run_surface --collect-only
    ;;
  *)
    echo "Expected PD_RHO_MODE in {canary,probe,probe-one,cell,cell-one,collect,collect-one}, got ${mode}." >&2
    exit 2
    ;;
esac
