#!/bin/bash -l
#SBATCH --job-name=conv2-rho-transfer
#SBATCH --account=umg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --constraint=v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00

set -euo pipefail

fail_value() {
  local expected="$1"
  local provided="$2"
  printf 'Expected %s. Provided value: %s.\n' "${expected}" "${provided}" >&2
  exit 2
}

required=(
  MNIST_CONV_SOURCE_REPO
  MNIST_CONV_TRANSFER_MANIFEST
  MNIST_CONV_SOURCE_STUDY
  MNIST_CONV_DATASET_ROOT
  MNIST_CONV_EXECUTION_ALLOCATION
)
for variable in "${required[@]}"; do
  [[ -n "${!variable:-}" ]] || fail_value "${variable} to be non-empty" "${!variable:-unset}"
done

[[ "${MNIST_CONV_EXECUTION_ALLOCATION}" == "AD011016471R1" ]] || \
  fail_value "MNIST_CONV_EXECUTION_ALLOCATION to equal AD011016471R1" "${MNIST_CONV_EXECUTION_ALLOCATION}"
[[ "${SLURM_JOB_ACCOUNT:-}" == "umg@v100" ]] || \
  fail_value "the live Slurm account to equal umg@v100" "${SLURM_JOB_ACCOUNT:-unset}"
[[ "${SLURM_JOB_PARTITION:-}" == "gpu_p13" ]] || \
  fail_value "the live Slurm partition to equal gpu_p13" "${SLURM_JOB_PARTITION:-unset}"
[[ "${SLURM_CPUS_PER_TASK:-}" == "16" ]] || \
  fail_value "SLURM_CPUS_PER_TASK to equal 16" "${SLURM_CPUS_PER_TASK:-unset}"

job_record="$(scontrol show job "${SLURM_JOB_ID}" -o)"
[[ "${job_record}" == *"QOS=qos_gpu-t3"* ]] || \
  fail_value "the live Slurm QOS to equal qos_gpu-t3" "${job_record}"
[[ "${job_record}" == *"v100-32g"* ]] || \
  fail_value "the live Slurm constraint to contain v100-32g" "${job_record}"

if [[ -x /gpfslocalsup/bin/idrenv ]]; then
  eval "$(/gpfslocalsup/bin/idrenv -d umg)"
else
  fail_value "the Jean Zay idrenv helper to exist" "/gpfslocalsup/bin/idrenv"
fi

export MODULESHOME=/lustre/fshomisc/sup/spack_soft/environment-modules/4.3.1/gcc-11.3.1-wf7m7j6whgecysm2fm5n73sm4jg7txup
export MODULEPATH=/lustre/fshomisc/sup/hpe/pub/module-rh/modulefiles:/lustre/fshomisc/sup/hpe/pub/modules-idris-env4/modulefiles/linux-rhel9-skylake_avx512
modulecmd_path="${MODULESHOME}/bin/modulecmd"
[[ -x "${modulecmd_path}" ]] || fail_value "modulecmd to be executable" "${modulecmd_path}"
eval "$("${modulecmd_path}" bash load pytorch-gpu/py3/2.5.0)"

python_bin="$(command -v python)"
[[ -n "${python_bin}" ]] || fail_value "the PyTorch module to provide python" "not found"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | xargs)"
gpu_memory="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | xargs)"
[[ "${gpu_name}" == *"V100"* ]] || fail_value "the allocated GPU to be a V100" "${gpu_name}"
[[ "${gpu_memory}" == "32768" ]] || fail_value "the allocated GPU memory to equal 32768 MiB" "${gpu_memory}"

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MNIST_CONV_V6_R3_AUTHORIZATION='AD010913993R3:fmu@v100:gpu_p13:qos_gpu-t3:v100-32g:gpu1'

transfer_root="$(dirname "${MNIST_CONV_TRANSFER_MANIFEST}")"
log_dir="${transfer_root}/slurm/${SLURM_JOB_ID}"
mkdir -p "${log_dir}"
receipt="${log_dir}/execution_receipt.json"
RECEIPT_PATH="${receipt}" GPU_NAME="${gpu_name}" GPU_MEMORY="${gpu_memory}" \
"${python_bin}" -c '
import json, os, pathlib
path = pathlib.Path(os.environ["RECEIPT_PATH"])
value = {
    "schema_version": "mnist-conv-lr-exploratory-transfer-execution/v1",
    "transfer_manifest": os.environ["MNIST_CONV_TRANSFER_MANIFEST"],
    "source_study": os.environ["MNIST_CONV_SOURCE_STUDY"],
    "allocation": os.environ["MNIST_CONV_EXECUTION_ALLOCATION"],
    "project": "umg",
    "slurm_account": os.environ["SLURM_JOB_ACCOUNT"],
    "partition": os.environ["SLURM_JOB_PARTITION"],
    "qos": "qos_gpu-t3",
    "constraint": "v100-32g",
    "gpu_name": os.environ["GPU_NAME"],
    "gpu_memory_mib": int(os.environ["GPU_MEMORY"]),
    "job_id": os.environ["SLURM_JOB_ID"],
    "concurrent_children": 2,
    "user_authorized": True,
    "canonical_v6_confirmation": False,
}
encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
if path.exists() and path.read_bytes() != encoded:
    raise SystemExit(f"conflicting immutable execution receipt: {path}")
path.write_bytes(encoded)
'

printf '[rho-transfer] account=%s job=%s gpu=%s memory_mib=%s children=2\n' \
  "${SLURM_JOB_ACCOUNT}" "${SLURM_JOB_ID}" "${gpu_name}" "${gpu_memory}"

cd "${MNIST_CONV_SOURCE_REPO}"
exec "${python_bin}" -m experiments.run_mnist_conv_lr_v6_exploratory_transfer run-pack \
  --manifest "${MNIST_CONV_TRANSFER_MANIFEST}" \
  --source-study "${MNIST_CONV_SOURCE_STUDY}" \
  --data-root "${MNIST_CONV_DATASET_ROOT}" \
  --device cuda \
  --python "${python_bin}" \
  --log-dir "${log_dir}"
