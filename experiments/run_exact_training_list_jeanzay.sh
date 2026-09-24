#!/usr/bin/env bash
# Submit after the same-config local semantic smoke. SOURCE LIST OUTPUT DATA.
set -euo pipefail
(( $# == 4 )) || exit 2
export MODULESHOME=/lustre/fshomisc/sup/spack_soft/environment-modules/4.3.1/gcc-11.3.1-wf7m7j6whgecysm2fm5n73sm4jg7txup
export MODULEPATH=/lustre/fshomisc/sup/hpe/pub/module-rh/modulefiles:/lustre/fshomisc/sup/hpe/pub/modules-idris-env4/modulefiles/linux-rhel9-skylake_avx512
module_command="$MODULESHOME/bin/modulecmd"
eval "$("$module_command" bash purge)"
eval "$("$module_command" bash load pytorch-gpu/py3/2.5.0)"
python_bin=$(command -v python)
exec bash "$1/experiments/run_exact_training_list.sh" "$1" "$2" "$3" "$4" jean-zay "$python_bin"
