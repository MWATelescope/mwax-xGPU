#!/usr/bin/env bash
#
# Build mwax-xGPU for each tiles value with the same make commands as the
# Ansible playbook.
#
# The script does NOT run "make install" and does not change /usr/local.
#
set -euo pipefail

# Build parameters copied from the Ansible playbook.
readonly NFREQUENCY=6400
readonly TILES_SPLIT=192
readonly NTIME_UP_TO_SPLIT=52
readonly NTIME_PIPE_UP_TO_SPLIT=52
readonly NTIME_OVER_SPLIT=56
readonly NTIME_PIPE_OVER_SPLIT=28        

# Directories
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src_dir="${script_dir}/src"
readonly out_dir="${script_dir}/build"
readonly cuda_dir="/usr/local/cuda"

# Compilation
readonly device=0
readonly arch="sm_86"
readonly host_cc="/usr/bin/gcc"

nvcc_cmd=("${cuda_dir}/bin/nvcc")
if [[ -n "${host_cc}" ]]; then
    nvcc_cmd+=(-ccbin "${host_cc}")
fi
readonly nvcc_cmd
readonly LFLAGS="-L${cuda_dir}/lib64 -L. -lrt -lm"

echo Cleaning up old build dir
rm -rfv ${out_dir}
mkdir ${out_dir}
# 16 32 48 64 80 96 112 128 144 160 184 200 208 224 240 256
for tiles in 200; do
    # Undo any previous mods: the Makefile edits xgpu_info.h in place.
    /usr/bin/git restore "${src_dir}/xgpu_info.h"

    echo "=== ${tiles}T / ${NFREQUENCY} ==="
    if (( tiles <= TILES_SPLIT )); then
        ntime=${NTIME_UP_TO_SPLIT}
        ntime_pipe=${NTIME_PIPE_UP_TO_SPLIT}
    else
        ntime=${NTIME_OVER_SPLIT}
        ntime_pipe=${NTIME_PIPE_OVER_SPLIT}
    fi

    label="${tiles}t_${NFREQUENCY}"
    cfg_out="${out_dir}/${label}"
    mkdir -p "${cfg_out}"

    pushd "${script_dir}/src" > /dev/null
        
    if ! { make clean NSTATION="${tiles}" \
        && make NVCC="${nvcc_cmd[*]}" LFLAGS="${LFLAGS}" CUDA_ARCH="${arch}" NSTATION="${tiles}" \
            NFREQUENCY="${NFREQUENCY}" NTIME="${ntime}" NTIME_PIPE="${ntime_pipe}" ; } \
        > "${cfg_out}/build.log" 2>&1; then
        echo "BUILD FAILED: see ${cfg_out}/build.log"
        echo "build=FAIL" > "${cfg_out}/results.txt"                
        continue
    else
        ./xgpuinfo > "${cfg_out}/xgpuinfo.txt"
        cat ${cfg_out}/xgpuinfo.txt
        lib="libxgpumwax${tiles}t_${NFREQUENCY}.so"        
        out_lib="${cfg_out}/${lib}"
        mv "${src_dir}/${lib}" "${out_lib}"
        cp "${src_dir}/xgpu_info.h" "${cfg_out}/xgpu_info.h"
        echo "Successfully built ${out_lib}"                
    fi

    popd > /dev/null
done