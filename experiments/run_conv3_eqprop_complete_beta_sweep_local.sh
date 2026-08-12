#!/usr/bin/env bash
set -uo pipefail

export KMP_DISABLE_SHM=1
export KMP_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

study_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
study_python=/home/filip/miniconda3/envs/py312/bin/python
source_bundle=results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/conv-current-commonbase
study_output=results/perfectdiode-conv3-eqprop-beta-1em3-to-1e2-init-best-true-dtype-seed0-20260811-v1
mode=${1:-production}
failed_cases=0

cd "$study_root" || exit 1
mkdir -p "$study_output"

run_audit() {
    "$@" || failed_cases=$((failed_cases + 1))
}

run_beta() {
    local estimator=$1
    local beta_label=$2
    local injected_beta=$3
    local baseline_beta=$4
    local ours_beta=$5
    local legacy_beta=$6
    local T=$7
    local K=$8
    local smoke_flag=${9:-}
    local estimator_label=${estimator/positive_one_sided/one-sided}

    run_audit "$study_python" experiments/audit_eqprop_float64_shadow.py \
        --config "$source_bundle/config.resolved.json" \
        --source-run "$source_bundle" \
        --case "conv3:baseline:reconstructed_initialization:$baseline_beta" \
        --case "conv3:baseline:best_validation:$baseline_beta" \
        --case "conv3:ours:reconstructed_initialization:$ours_beta" \
        --case "conv3:ours:best_validation:$ours_beta" \
        --case "conv3:legacy:reconstructed_initialization:$legacy_beta" \
        --case "conv3:legacy:best_validation:$legacy_beta" \
        --eqprop-variant "$estimator" \
        --explicit-beta-label complete_injected_beta_sweep \
        --tk-override "$T" "$K" \
        --allow-out-of-grid-beta \
        --batch-index 0 \
        --output-root "$study_output" \
        --run-id "${smoke_flag:+${smoke_flag}-}${estimator_label}-B${beta_label}-tk${T}" \
        --device cuda \
        --target local:RTX3090 \
        ${smoke_flag:+--smoke}
}

if [[ "$mode" == "smoke" ]]; then
    run_beta centered 1em1 0.1 0.1 0.0015625 0.0000244140625 2 2 smoke-attempt-02
    exit "$failed_cases"
fi

if [[ "$mode" == "smoke-stress" ]]; then
    run_beta centered 100 100 100 1.5625 0.0244140625 2 2 smoke-stress-attempt-01
    exit "$failed_cases"
fi

if [[ "$mode" != "production" ]]; then
    echo "usage: $0 [smoke|smoke-stress|production]" >&2
    exit 2
fi

for estimator in positive_one_sided centered; do
    while IFS=: read -r beta_label injected_beta baseline_beta ours_beta legacy_beta; do
        run_beta "$estimator" "$beta_label" "$injected_beta" "$baseline_beta" "$ours_beta" "$legacy_beta" 64 64
    done <<'EOF'
1em3:0.001:0.001:0.000015625:0.000000244140625
3em3:0.003:0.003:0.000046875:0.000000732421875
1em2:0.01:0.01:0.00015625:0.00000244140625
3em2:0.03:0.03:0.00046875:0.00000732421875
1em1:0.1:0.1:0.0015625:0.0000244140625
3em1:0.3:0.3:0.0046875:0.0000732421875
1:1:1:0.015625:0.000244140625
3:3:3:0.046875:0.000732421875
10:10:10:0.15625:0.00244140625
30:30:30:0.46875:0.00732421875
100:100:100:1.5625:0.0244140625
EOF
done

exit "$failed_cases"
