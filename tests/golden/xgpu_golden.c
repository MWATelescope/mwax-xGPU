// xgpu_golden: regression ("golden") test harness for the mwax-xGPU library.
//
// The harness runs a fixed set of test cases against one build of the
// library and writes the results as "key=value" lines. Run it on the old
// build (before a change) and on the new build (after a change), then use
// compare_golden.py to compare the two result files.
//
// Each case checks the GPU output against the CPU reference
// (xgpuOmpXengine). Integer-valued inputs give exact float results, so these
// cases require an exact match and an identical output hash.
//
// MWA mods: xgpuCudaXengine() does not copy input to the GPU. The caller must
// write the input into the device buffers array_d[0] and array_d[1]. This
// harness does the same as the production code, through the
// XGPUInternalPrefix struct below.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <cuda_runtime_api.h>

#include "xgpu.h"

#if defined(FIXED_POINT) || defined(DP4A)
#error "xgpu_golden supports only the FP32 build (no FIXED_POINT, no DP4A)"
#endif

// Mirror of the first members of XGPUInternalContext in cuda_xengine.cu.
// The layout of these members must not change.
typedef struct XGPUInternalPrefixStruct {
  int device;
  ComplexInput *array_d[2];
} XGPUInternalPrefix;

// Default PRNG seed. Override with -s.
#define DEFAULT_SEED 20261027ULL

// Maximum absolute input value for the integer cases.
#define INT_SMALL_MAX_ABS 7
#define INT_LARGE_MAX_ABS 127

// Number of xgpuCudaXengine() calls in the accumulation case.
#define ACCUMULATE_CALLS 3

// Maximum error for the non-integer float case (GPU vs CPU), as a fraction of
// the largest CPU output magnitude. A per-element relative error is not used:
// elements near zero have a large relative error from float32 rounding alone.
#define FLOAT_TOLERANCE 1.0e-5

// xGPU requires NSTATION to be a multiple of this value.
#define NSTATION_MULTIPLE 16

// Number of calls for the timing case.
#define TIMING_WARMUP_CALLS 3
#define TIMING_CALLS 20

// The MWA mods support at most two pipeline stages (array_d[0] and [1]).
#define MAX_PIPE_LENGTH 2

// FNV-1a 64-bit hash parameters.
#define FNV1A_OFFSET_BASIS 0xcbf29ce484222325ULL
#define FNV1A_PRIME 0x100000001b3ULL

// splitmix64 constants.
#define SPLITMIX_INCREMENT 0x9e3779b97f4a7c15ULL
#define SPLITMIX_MULT_1 0xbf58476d1ce4e5b9ULL
#define SPLITMIX_MULT_2 0x94d049bb133111ebULL
#define SPLITMIX_SHIFT_1 30
#define SPLITMIX_SHIFT_2 27
#define SPLITMIX_SHIFT_3 31

// Bits used to make a float in [0, 1) with an exact 24-bit mantissa.
#define FLOAT_MANTISSA_BITS 24
#define UINT64_BITS 64

#define NSEC_PER_MSEC 1.0e6
#define MSEC_PER_SEC 1.0e3

#define STATUS_PASS "PASS"
#define STATUS_FAIL "FAIL"

static int g_fail_count = 0;
static FILE *g_out = NULL;

static uint64_t splitmix64(uint64_t *state) {
  uint64_t z = (*state += SPLITMIX_INCREMENT);
  z = (z ^ (z >> SPLITMIX_SHIFT_1)) * SPLITMIX_MULT_1;
  z = (z ^ (z >> SPLITMIX_SHIFT_2)) * SPLITMIX_MULT_2;
  return z ^ (z >> SPLITMIX_SHIFT_3);
}

static uint64_t fnv1a64(const void *data, size_t nbytes) {
  const unsigned char *p = (const unsigned char *)data;
  uint64_t h = FNV1A_OFFSET_BASIS;
  for (size_t i = 0; i < nbytes; i++) {
    h ^= p[i];
    h *= FNV1A_PRIME;
  }
  return h;
}

static void fill_zero(ComplexInput *in, size_t n) { memset(in, 0, n * sizeof(ComplexInput)); }

static void fill_int(ComplexInput *in, size_t n, int max_abs, uint64_t *state) {
  const uint64_t span = (uint64_t)(2 * max_abs + 1);
  for (size_t i = 0; i < n; i++) {
    in[i].real = (float)((int)(splitmix64(state) % span) - max_abs);
    in[i].imag = (float)((int)(splitmix64(state) % span) - max_abs);
  }
}

// Uniform values in [-1, 1) with an exact 24-bit mantissa.
static float uniform_pm1(uint64_t *state) {
  uint64_t bits = splitmix64(state) >> (UINT64_BITS - FLOAT_MANTISSA_BITS);
  float u = (float)bits / (float)(1ULL << FLOAT_MANTISSA_BITS);
  return 2.0f * u - 1.0f;
}

static void fill_float(ComplexInput *in, size_t n, uint64_t *state) {
  for (size_t i = 0; i < n; i++) {
    in[i].real = uniform_pm1(state);
    in[i].imag = uniform_pm1(state);
  }
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * MSEC_PER_SEC + (double)ts.tv_nsec / NSEC_PER_MSEC;
}

static void record_status(const char *name, int ok) {
  fprintf(g_out, "case.%s.status=%s\n", name, ok ? STATUS_PASS : STATUS_FAIL);
  printf("%-12s %s\n", name, ok ? STATUS_PASS : STATUS_FAIL);
  if (!ok) {
    g_fail_count++;
  }
}

static int check_cuda(const char *where) {
  cudaError_t err = cudaDeviceSynchronize();
  if (err == cudaSuccess) {
    err = cudaGetLastError();
  }
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error after %s: %s\n", where, cudaGetErrorString(err));
    return 0;
  }
  return 1;
}

// Copy the full input vector to the device. The two device buffers are
// contiguous (MWA mod), so one copy fills array_d[0] and array_d[1].
static int upload(const XGPUContext *ctx, const XGPUInfo *info) {
  const XGPUInternalPrefix *prefix = (const XGPUInternalPrefix *)ctx->internal;
  cudaError_t err = cudaMemcpy(prefix->array_d[0], ctx->array_h, info->vecLength * sizeof(ComplexInput),
                               cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy to array_d failed: %s\n", cudaGetErrorString(err));
    return 0;
  }
  return check_cuda("upload");
}

static int xengine(XGPUContext *ctx, int sync_op) {
  int rc = xgpuCudaXengine(ctx, sync_op);
  if (rc != XGPU_OK) {
    fprintf(stderr, "xgpuCudaXengine returned %d\n", rc);
    return 0;
  }
  return check_cuda("xgpuCudaXengine");
}

static int clear(XGPUContext *ctx) {
  int rc = xgpuClearDeviceIntegrationBuffer(ctx);
  if (rc != XGPU_OK) {
    fprintf(stderr, "xgpuClearDeviceIntegrationBuffer returned %d\n", rc);
    return 0;
  }
  return check_cuda("xgpuClearDeviceIntegrationBuffer");
}

// Hash the raw (register tile order) output and record the hash.
static uint64_t record_hash(const char *name, const XGPUContext *ctx, const XGPUInfo *info) {
  uint64_t h = fnv1a64(ctx->matrix_h, info->matLength * sizeof(Complex));
  fprintf(g_out, "case.%s.hash=%016llx\n", name, (unsigned long long)h);
  return h;
}

// Compare reordered GPU output with scale * CPU output. If tolerance < 0 the
// match must be exact. Else the error of each element must be at most
// tolerance times the largest magnitude in scale * CPU output.
static int compare_cpu(const char *name, const Complex *gpu, const Complex *cpu, size_t n, float scale,
                       double tolerance) {
  double max_mag = 0.0;
  for (size_t i = 0; i < n; i++) {
    double er = (double)scale * cpu[i].real;
    double ei = (double)scale * cpu[i].imag;
    double mag = sqrt(er * er + ei * ei);
    if (mag > max_mag) {
      max_mag = mag;
    }
  }

  size_t mismatches = 0;
  double max_abs = 0.0;
  double max_norm = 0.0;
  for (size_t i = 0; i < n; i++) {
    double er = (double)scale * cpu[i].real;
    double ei = (double)scale * cpu[i].imag;
    double dr = (double)gpu[i].real - er;
    double di = (double)gpu[i].imag - ei;
    double abs_err = sqrt(dr * dr + di * di);
    double norm_err = (max_mag == 0.0) ? abs_err : abs_err / max_mag;
    if (abs_err > max_abs) {
      max_abs = abs_err;
    }
    if (norm_err > max_norm) {
      max_norm = norm_err;
    }
    if (tolerance < 0.0) {
      if (gpu[i].real != (float)er || gpu[i].imag != (float)ei) {
        mismatches++;
      }
    } else if (norm_err > tolerance) {
      mismatches++;
    }
  }
  fprintf(g_out, "case.%s.mismatches=%zu\n", name, mismatches);
  fprintf(g_out, "case.%s.max_abs_err=%.9g\n", name, max_abs);
  fprintf(g_out, "case.%s.max_norm_err=%.9g\n", name, max_norm);
  return mismatches == 0;
}

// Run "calls" correlations of the current host input without clearing in
// between, and check the result against "calls" times the CPU reference.
// If new_input is 0, the input and the CPU reference from the previous case
// are used again. The output hash is returned in *hash.
static int run_and_check(const char *name, XGPUContext *ctx, const XGPUInfo *info, Complex *cpu, int new_input,
                         int calls, double tolerance, uint64_t *hash) {
  if (new_input) {
    if (!upload(ctx, info)) {
      return 0;
    }
    xgpuOmpXengine(cpu, ctx->array_h);
  }
  if (!clear(ctx)) {
    return 0;
  }
  for (int c = 0; c < calls - 1; c++) {
    if (!xengine(ctx, SYNCOP_SYNC_COMPUTE)) {
      return 0;
    }
  }
  if (!xengine(ctx, SYNCOP_DUMP)) {
    return 0;
  }
  *hash = record_hash(name, ctx, info);
  xgpuReorderMatrix(ctx->matrix_h);
  return compare_cpu(name, ctx->matrix_h, cpu, info->triLength, (float)calls, tolerance);
}

static void write_info(const XGPUInfo *info) {
  fprintf(g_out, "info.version=%s\n", xgpuVersionString());
  fprintf(g_out, "info.npol=%u\n", info->npol);
  fprintf(g_out, "info.nstation=%u\n", info->nstation);
  fprintf(g_out, "info.nbaseline=%u\n", info->nbaseline);
  fprintf(g_out, "info.nfrequency=%u\n", info->nfrequency);
  fprintf(g_out, "info.ntime=%u\n", info->ntime);
  fprintf(g_out, "info.ntimepipe=%u\n", info->ntimepipe);
  fprintf(g_out, "info.input_type=%u\n", info->input_type);
  fprintf(g_out, "info.compute_type=%u\n", info->compute_type);
  fprintf(g_out, "info.vecLength=%llu\n", info->vecLength);
  fprintf(g_out, "info.vecLengthPipe=%llu\n", info->vecLengthPipe);
  fprintf(g_out, "info.matLength=%llu\n", info->matLength);
  fprintf(g_out, "info.triLength=%llu\n", info->triLength);
  fprintf(g_out, "info.matrix_order=%u\n", info->matrix_order);
  fprintf(g_out, "info.shared_atomic_size=%zu\n", info->shared_atomic_size);
  fprintf(g_out, "info.complex_block_size=%zu\n", info->complex_block_size);
}

static void usage(const char *prog) {
  fprintf(stderr, "Usage: %s -d <device> -o <results_file> [-s <seed>]\n", prog);
}

int main(int argc, char **argv) {
  int device = -1;
  const char *out_path = NULL;
  uint64_t seed = DEFAULT_SEED;
  int opt;

  while ((opt = getopt(argc, argv, "d:o:s:h")) != -1) {
    switch (opt) {
      case 'd':
        device = atoi(optarg);
        break;
      case 'o':
        out_path = optarg;
        break;
      case 's':
        seed = strtoull(optarg, NULL, 0);
        break;
      default:
        usage(argv[0]);
        return EXIT_FAILURE;
    }
  }
  if (device < 0 || out_path == NULL) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  g_out = fopen(out_path, "w");
  if (!g_out) {
    perror(out_path);
    return EXIT_FAILURE;
  }

  XGPUInfo info;
  xgpuInfo(&info);
  write_info(&info);
  fprintf(g_out, "run.seed=%llu\n", (unsigned long long)seed);

  if (info.nstation % NSTATION_MULTIPLE != 0) {
    fprintf(stderr, "NSTATION %u is not a multiple of %d: not a valid xGPU configuration\n", info.nstation,
            NSTATION_MULTIPLE);
    fclose(g_out);
    return EXIT_FAILURE;
  }

  if (info.ntime / info.ntimepipe > MAX_PIPE_LENGTH) {
    fprintf(stderr, "PIPE_LENGTH %u > %d is not supported by the MWA mods\n", info.ntime / info.ntimepipe,
            MAX_PIPE_LENGTH);
    fclose(g_out);
    return EXIT_FAILURE;
  }

  XGPUContext ctx;
  memset(&ctx, 0, sizeof(ctx));
  int rc = xgpuInit(&ctx, device);
  if (rc != XGPU_OK) {
    fprintf(stderr, "xgpuInit returned %d\n", rc);
    fclose(g_out);
    return EXIT_FAILURE;
  }

  Complex *cpu = (Complex *)malloc(info.triLength * sizeof(Complex));
  if (!cpu) {
    fprintf(stderr, "Cannot allocate CPU reference buffer\n");
    xgpuFree(&ctx);
    fclose(g_out);
    return EXIT_FAILURE;
  }

  uint64_t state = seed;
  uint64_t hash_single = 0;
  uint64_t hash_other = 0;
  int ok;

  // Case: all-zero input gives all-zero output.
  fill_zero(ctx.array_h, info.vecLength);
  ok = run_and_check("zero", &ctx, &info, cpu, 1, 1, -1.0, &hash_other);
  record_status("zero", ok);

  // Case: small integers, exact match.
  fill_int(ctx.array_h, info.vecLength, INT_SMALL_MAX_ABS, &state);
  ok = run_and_check("int_small", &ctx, &info, cpu, 1, 1, -1.0, &hash_single);
  record_status("int_small", ok);

  // Case: same input, several calls without clear, exact sum.
  ok = run_and_check("accumulate", &ctx, &info, cpu, 0, ACCUMULATE_CALLS, -1.0, &hash_other);
  record_status("accumulate", ok);

  // Case: clear after accumulation gives the single-call result again.
  ok = run_and_check("clear", &ctx, &info, cpu, 0, 1, -1.0, &hash_other);
  record_status("clear", ok && hash_other == hash_single);

  // Case: full 8-bit integer range, exact match.
  fill_int(ctx.array_h, info.vecLength, INT_LARGE_MAX_ABS, &state);
  ok = run_and_check("int_large", &ctx, &info, cpu, 1, 1, -1.0, &hash_other);
  record_status("int_large", ok);

  // Case: non-integer floats, match within tolerance.
  fill_float(ctx.array_h, info.vecLength, &state);
  ok = run_and_check("float", &ctx, &info, cpu, 1, 1, FLOAT_TOLERANCE, &hash_other);
  record_status("float", ok);

  // Timing: kernel time per call, input already on the device.
  ok = clear(&ctx);
  for (int i = 0; ok && i < TIMING_WARMUP_CALLS; i++) {
    ok = xengine(&ctx, SYNCOP_SYNC_COMPUTE);
  }
  double t_min = 0.0, t_max = 0.0, t_sum = 0.0;
  for (int i = 0; ok && i < TIMING_CALLS; i++) {
    double t0 = now_ms();
    ok = xengine(&ctx, SYNCOP_SYNC_COMPUTE);
    double dt = now_ms() - t0;
    t_sum += dt;
    if (i == 0 || dt < t_min) {
      t_min = dt;
    }
    if (i == 0 || dt > t_max) {
      t_max = dt;
    }
  }
  if (ok) {
    fprintf(g_out, "timing.calls=%d\n", TIMING_CALLS);
    fprintf(g_out, "timing.mean_ms=%.4f\n", t_sum / TIMING_CALLS);
    fprintf(g_out, "timing.min_ms=%.4f\n", t_min);
    fprintf(g_out, "timing.max_ms=%.4f\n", t_max);
  }
  record_status("timing", ok);

  fprintf(g_out, "summary.failures=%d\n", g_fail_count);
  printf("Failures: %d\n", g_fail_count);

  free(cpu);
  xgpuFree(&ctx);
  fclose(g_out);
  return g_fail_count == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
