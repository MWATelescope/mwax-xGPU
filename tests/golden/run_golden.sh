#!/usr/bin/env bash
#
# Build mwax-xGPU for each tiles value with the same make commands as the
# Ansible playbook, then run the golden test harness on each build.
#
# The script does NOT run "make install" and does not change /usr/local.
# It uses the GPU, so stop the correlator on the server first.
#
# Example:
#   ./run_golden.sh -r https://github.com/MWATelescope/mwax-xGPU.git -b master \
#       -c /usr/local/cuda -g gcc-11 -a sm_86 -d 0 -t "32 64 128 256" -o ~/golden/before
#
set -euo pipefail

# Build parameters copied from the Ansible playbook.
readonly NFREQUENCY=6400
readonly TILES_SPLIT=192
readonly NTIME_UP_TO_SPLIT=52
readonly NTIME_PIPE_UP_TO_SPLIT=52
readonly NTIME_OVER_SPLIT=56
readonly NTIME_PIPE_OVER_SPLIT=28

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly HARNESS_SRC="${SCRIPT_DIR}/xgpu_golden.c"
readonly HARNESS_BIN="xgpu_golden"

usage() {
    cat >&2 <<EOF
Usage: $0 -r <repo> -b <branch|commit> -c <cuda_dir> -g <host_cc> -a <sm_arch> -d <gpu_device> -t "<tiles list>" -o <out_dir>
  -r  git repo URL or local path to build the library from
  -b  branch, tag or commit to build
  -c  CUDA toolkit directory (e.g. /usr/local/cuda)
  -g  host compiler for nvcc -ccbin (e.g. gcc-11)
  -a  GPU architecture (e.g. sm_86)
  -d  GPU device index
  -t  space-separated list of tiles values
  -o  output directory (must not exist)
EOF
    exit 1
}

repo="" ref="" cuda_dir="" host_cc="" arch="" device="" tiles_list="" out_dir=""
while getopts "r:b:c:g:a:d:t:o:h" opt; do
    case "${opt}" in
        r) repo="${OPTARG}" ;;
        b) ref="${OPTARG}" ;;
        c) cuda_dir="${OPTARG}" ;;
        g) host_cc="${OPTARG}" ;;
        a) arch="${OPTARG}" ;;
        d) device="${OPTARG}" ;;
        t) tiles_list="${OPTARG}" ;;
        o) out_dir="${OPTARG}" ;;
        *) usage ;;
    esac
done
for v in repo ref cuda_dir host_cc arch device tiles_list out_dir; do
    if [[ -z "${!v}" ]]; then
        echo "Missing argument: ${v}" >&2
        usage
    fi
done
if [[ -e "${out_dir}" ]]; then
    echo "Output directory exists: ${out_dir}" >&2
    exit 1
fi

nvcc_cmd=("${cuda_dir}/bin/nvcc" -ccbin "${host_cc}")
readonly nvcc_cmd
readonly LFLAGS="-L${cuda_dir}/lib64 -L. -lrt -lm"

mkdir -p "${out_dir}"
out_dir="$(cd "${out_dir}" && pwd)"
work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

{
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname)"
    echo "repo=${repo}"
    echo "ref=${ref}"
    echo "arch=${arch}"
    echo "host_cc=${host_cc}"
    echo "nvcc_version=$("${cuda_dir}/bin/nvcc" --version | tail -n 1)"
    echo "host_cc_version=$("${host_cc}" --version | head -n 1)"
    echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "${device}")"
    echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${device}")"
} > "${out_dir}/toolchain.txt"

overall=0
for tiles in ${tiles_list}; do
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
    src_dir="${work_dir}/${label}"
    mkdir -p "${cfg_out}"

    # Fresh clone per tiles value: the Makefile edits xgpu_info.h in place.
    git clone -q "${repo}" "${src_dir}"
    git -C "${src_dir}" checkout -q "${ref}"
    git -C "${src_dir}" rev-parse HEAD > "${cfg_out}/commit.txt"

    pushd "${src_dir}/src" > /dev/null
    if ! { make clean NSTATION="${tiles}" \
        && make NVCC="${nvcc_cmd[*]}" LFLAGS="${LFLAGS}" CUDA_ARCH="${arch}" NSTATION="${tiles}" \
            NFREQUENCY="${NFREQUENCY}" NTIME="${ntime}" NTIME_PIPE="${ntime_pipe}"; } \
        > "${cfg_out}/build.log" 2>&1; then
        echo "BUILD FAILED: see ${cfg_out}/build.log"
        echo "build=FAIL" > "${cfg_out}/results.txt"
        overall=1
        popd > /dev/null
        continue
    fi

    lib="libxgpumwax${tiles}t_${NFREQUENCY}.so"
    # shellcheck disable=SC2016  # $ORIGIN is for the linker, not the shell.
    if ! "${nvcc_cmd[@]}" -O2 -arch="${arch}" -I. -o "${HARNESS_BIN}" "${HARNESS_SRC}" \
        -L. -l"xgpumwax${tiles}t_${NFREQUENCY}" -Xlinker -rpath,'$ORIGIN' \
        >> "${cfg_out}/build.log" 2>&1; then
        echo "HARNESS BUILD FAILED: see ${cfg_out}/build.log"
        echo "build=FAIL" > "${cfg_out}/results.txt"
        overall=1
        popd > /dev/null
        continue
    fi

    ./xgpuinfo > "${cfg_out}/xgpuinfo.txt"
    nm -D --defined-only "${lib}" | awk '{print $3}' | grep '^xgpu' | sort > "${cfg_out}/exports.txt"

    if ! ./"${HARNESS_BIN}" -d "${device}" -o "${cfg_out}/results.txt" | tee "${cfg_out}/run.log"; then
        overall=1
    fi
    popd > /dev/null
    rm -rf "${src_dir}"
done

echo "Results in ${out_dir}"
exit "${overall}"
