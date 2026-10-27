# Golden tests

Regression tests for the FP32 build of mwax-xGPU (no `FIXED_POINT`, no `DP4A`). Run them on the old build, then on the
new build, then compare.

## Files

- `xgpu_golden.c`: test harness. Links against one library build and writes `results.txt`.
- `run_golden.sh`: builds the library for each tiles value with the Ansible make commands, then runs the harness.
  It does not run `make install`.
- `compare_golden.py`: compares two result directories.

## Test cases

| Case         | Input                       | Check                                              |
|--------------|-----------------------------|----------------------------------------------------|
| `zero`       | all zeros                   | exact match with CPU (all zero)                    |
| `int_small`  | integers -7..7              | exact match with CPU                               |
| `accumulate` | same input, 3 calls         | exact match with 3 x CPU                           |
| `clear`      | same input after clear      | exact match with CPU, same hash as `int_small`     |
| `int_large`  | integers -127..127          | exact match with CPU                               |
| `float`      | uniform floats in [-1, 1)   | error vs CPU <= 1e-5 of the largest output value   |
| `timing`     | `float` input               | mean time per `xgpuCudaXengine()` call             |

The tiles value must be a multiple of 16 (an xGPU requirement); the harness refuses other values.

Integer-valued inputs give exact float sums, so the output hash of these cases must not change between builds.

## Usage

Stop the correlator first. The tests use the GPU.

```
# Before (CUDA 11.8, master). CUDA 11.8 needs gcc-11 on Ubuntu 24.04.
./run_golden.sh -r https://github.com/MWATelescope/mwax-xGPU.git -b master \
    -c /usr/local/cuda-11.8 -g gcc-11 -a sm_86 -d 0 -t "<tiles list>" -o ~/golden/before

# After, step 1: the port on the SAME toolkit (CUDA 11.8). This isolates the port.
./run_golden.sh -r https://github.com/MWATelescope/mwax-xGPU.git -b cuda_13_update \
    -c /usr/local/cuda-11.8 -g gcc-11 -a sm_86 -d 0 -t "<tiles list>" -o ~/golden/after_118

# After, step 2: the port on CUDA 13.4. Do NOT use -g: CUDA 13 works with the default gcc.
./run_golden.sh -r https://github.com/MWATelescope/mwax-xGPU.git -b cuda_13_update \
    -c /usr/local/cuda-13.4 -a sm_86 -d 0 -t "<tiles list>" -o ~/golden/after_134

./compare_golden.py ~/golden/before ~/golden/after_118
./compare_golden.py ~/golden/before ~/golden/after_134
```

Include at least one tiles value above 192. Those builds use two pipeline stages (`array_d[1]`).

## Resources

At 256 tiles: about 22 GB host RAM, about 9 GB GPU memory. The CPU reference takes some minutes per input case.
