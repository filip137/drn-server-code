#!/usr/bin/env bash
set -euo pipefail

cd "${MNIST_CONV_REPO_ROOT:?}"
if [[ -n "${MNIST_CONV_MODULE:-}" ]]; then
  module purge
  module load "${MNIST_CONV_MODULE}"
fi
if [[ -n "${MNIST_CONV_IDRENV_PROJECT:-}" ]] && command -v idrenv >/dev/null 2>&1; then
  eval "$(idrenv -d "${MNIST_CONV_IDRENV_PROJECT}")"
fi

if [[ -n "${MNIST_CONV_R3_ALLOCATION:-}" ]]; then
  if [[ "${MNIST_CONV_R3_ALLOCATION}" != "AD010913993R3" || "${MNIST_CONV_R3_ALLOCATION}" != *R3 ]]; then
    printf 'Expected MNIST_CONV_R3_ALLOCATION to be AD010913993R3 and end in R3. Provided value: %s.\n' "${MNIST_CONV_R3_ALLOCATION}" >&2
    exit 2
  fi
  if [[ "${MNIST_CONV_THREADS_PER_PROCESS:-}" != "1" ]]; then
    printf 'Expected MNIST_CONV_THREADS_PER_PROCESS to equal 1. Provided value: %s.\n' "${MNIST_CONV_THREADS_PER_PROCESS:-<unset>}" >&2
    exit 2
  fi
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  for contract in \
    "SLURM_JOB_ACCOUNT:MNIST_CONV_EXPECTED_SLURM_ACCOUNT" \
    "SLURM_JOB_PARTITION:MNIST_CONV_EXPECTED_SLURM_PARTITION" \
    "SLURM_JOB_QOS:MNIST_CONV_EXPECTED_SLURM_QOS"
  do
    observed_name="${contract%%:*}"
    expected_name="${contract##*:}"
    observed="${!observed_name:-}"
    expected="${!expected_name:-}"
    if [[ -z "${expected}" || "${observed}" != "${expected}" ]]; then
      printf 'Expected %s to equal %s. Provided value: %s.\n' "${observed_name}" "${expected:-<unset>}" "${observed:-<unset>}" >&2
      exit 2
    fi
  done

  expected_constraint="${MNIST_CONV_EXPECTED_SLURM_CONSTRAINT:-}"
  observed_constraint="${SLURM_JOB_CONSTRAINTS:-}"
  if [[ -z "${expected_constraint}" ]]; then
    printf 'Expected MNIST_CONV_EXPECTED_SLURM_CONSTRAINT to be set. Provided value: <unset>.\n' >&2
    exit 2
  fi
  if [[ -z "${observed_constraint}" ]]; then
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
      printf 'Expected SLURM_JOB_ID to query a missing SLURM_JOB_CONSTRAINTS value. Provided value: <unset>.\n' >&2
      exit 2
    fi
    if ! constraint_rows="$(sacct -n -X -j "${SLURM_JOB_ID}" --format=Constraints --parsable2)"; then
      printf 'Expected sacct to resolve a missing SLURM_JOB_CONSTRAINTS value for job %s.\n' "${SLURM_JOB_ID}" >&2
      exit 2
    fi
    while IFS='|' read -r candidate_constraint _; do
      candidate_constraint="${candidate_constraint//[[:space:]]/}"
      if [[ -z "${candidate_constraint}" ]]; then
        continue
      fi
      if [[ -n "${observed_constraint}" && "${candidate_constraint}" != "${observed_constraint}" ]]; then
        printf 'Expected sacct to report one consistent Slurm constraint. Provided values: %s and %s.\n' "${observed_constraint}" "${candidate_constraint}" >&2
        exit 2
      fi
      observed_constraint="${candidate_constraint}"
    done <<< "${constraint_rows}"
  fi
  if [[ "${observed_constraint}" != "${expected_constraint}" ]]; then
    printf 'Expected the effective Slurm constraint to equal %s. Provided value: %s.\n' "${expected_constraint}" "${observed_constraint:-<unset>}" >&2
    exit 2
  fi
fi

exec "${MNIST_CONV_PYTHON:?}" -m experiments.mnist_conv lr-study \
  --stage "${MNIST_CONV_LR_STAGE:?}" \
  --study "${MNIST_CONV_LR_STUDY:?}" \
  --manifest "${MNIST_CONV_LR_MANIFEST:?}" \
  --entry-index "${SLURM_ARRAY_TASK_ID:?}" \
  --data-root "${MNIST_CONV_DATASET_ROOT:?}" \
  --device "${MNIST_CONV_DEVICE:?}"
