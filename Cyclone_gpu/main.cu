/*
 * Cyclone_gpu — GPU Bitcoin Puzzle Solver
 * Optimized batch algorithm ported from Cyclone_avx2:
 *   - Host-side GTable precomputation (8192 points)
 *   - One CUDA block = one batch of 512 keys (256 forward + 256 backward)
 *   - Per-thread Fermat modular inversion (fully parallel across 256 threads)
 *   - SHA-256 + RIPEMD-160 on GPU
 *
 * Build:
 *   nvcc -O3 -arch=sm_70 -std=c++14 -o Cyclone_gpu main.cu p2pkh_decoder.cpp
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#include "gpu_uint256.cuh"
#include "gpu_secp256k1.cuh"
#include "gpu_hash.cuh"
#include "p2pkh_decoder.h"

// ============================================================
// Tuneable constants
// ============================================================

// Number of 512-key batches processed per block per kernel launch.
// Each batch covers ~510 unique keys: 255 forward (priv+1..priv+255),
// 255 backward (priv-255..priv-1), plus the base point (priv).
// Higher values amortize per-block setup cost; lower values allow
// finer-grained status updates.
#ifndef BATCHES_PER_BLOCK
#  define BATCHES_PER_BLOCK 8
#endif

// Threads per block (= BATCH_SIZE; each thread handles 1 forward + 1 backward key,
// except tid==0 which handles only the base point; total ~510 keys per batch)
#define BLOCK_THREADS 256

// Grid width — overridden at runtime from GPU properties
#define DEFAULT_GRID_BLOCKS 2048

// GTable size: 32 rows × 256 entries = 8192 affine points
#define GTABLE_SIZE (32 * 256)

// ============================================================
// Device data
// ============================================================

// GTable in device global memory (524 KB)
static GPoint *d_gtable = nullptr;

// Plus / minus precomputed points in device global memory
static GPoint *d_plus  = nullptr;
static GPoint *d_minus = nullptr;

// 510*G (used to advance base within each block's batch loop)
static GPoint *d_step510G = nullptr;

// ============================================================
// Result structures
// ============================================================

struct FoundResult {
    int      flag;          // 1 = match found
    uint256_t priv_key;     // private key
    uint8_t  pubkey[33];    // compressed public key
    uint8_t  hash160[20];   // Hash160 of pubkey
};

struct CandidateEntry {
    uint256_t priv_key;
    uint8_t   pubkey[33];
    uint8_t   hash160[20];
};

static FoundResult    *d_result     = nullptr;
static CandidateEntry *d_candidates = nullptr;
static int            *d_cand_count = nullptr;
static uint64_t       *d_checked    = nullptr;

// ============================================================
// Utility: 256-bit addition on device by small integer step
// ============================================================

__device__ __forceinline__ void u256_add_small(uint256_t *r,
                                                const uint256_t *a,
                                                uint32_t step)
{
    // r = a + step
    uint64_t s = a->d[0] + step;
    uint64_t c = (s < a->d[0]) ? 1ULL : 0ULL;
    r->d[0] = s;
    s = a->d[1] + c; c = (s < a->d[1]) ? 1ULL : 0ULL; r->d[1] = s;
    s = a->d[2] + c; c = (s < a->d[2]) ? 1ULL : 0ULL; r->d[2] = s;
    r->d[3] = a->d[3] + c;
}

// 256-bit addition: r = a + b
__device__ __forceinline__ void u256_add_device(uint256_t *r,
                                                  const uint256_t *a,
                                                  const uint256_t *b)
{
    typedef unsigned __int128 u128;
    uint64_t carry = 0;
    u128 acc;
    acc = (u128)a->d[0] + b->d[0]; r->d[0] = (uint64_t)acc; carry = (uint64_t)(acc>>64);
    acc = (u128)a->d[1] + b->d[1] + carry; r->d[1] = (uint64_t)acc; carry = (uint64_t)(acc>>64);
    acc = (u128)a->d[2] + b->d[2] + carry; r->d[2] = (uint64_t)acc; carry = (uint64_t)(acc>>64);
    acc = (u128)a->d[3] + b->d[3] + carry; r->d[3] = (uint64_t)acc;
}

// Compare r = a >= b
__device__ __forceinline__ int u256_ge(const uint256_t *a, const uint256_t *b) {
    return u256_cmp(a, b) >= 0;
}

// ============================================================
// Kernel: batch search
//
// Grid:  gridDim.x = num_blocks
// Block: blockDim.x = BLOCK_THREADS (= 256)
//
// Each block processes BATCHES_PER_BLOCK × 510 keys:
//   - First, compute base = block_start * G  using GTable (thread 0)
//   - Then for each batch (~510 keys: 255 fwd + 255 bwd + 1 base):
//       * Thread tid computes deltaX[tid] = plus[tid].x - base.x
//       * Thread tid inverts deltaX[tid] via Fermat  (fully parallel)
//       * Thread tid computes 2 affine points (forward and backward)
//       * Thread tid computes Hash160 for each and checks against target
//       * Thread 0 advances base += step510G
//   - Repeat for next batch
// ============================================================

__global__ void batch_search_kernel(
    const GPoint * __restrict__ d_gtbl,
    const GPoint * __restrict__ d_pl,       // plus[256]
    const GPoint * __restrict__ d_mn,       // minus[256]
    const GPoint * __restrict__ d_step,     // pointer to step510G
    const uint8_t* __restrict__ target_h160,
    uint256_t      range_start,
    uint256_t      range_end,
    uint64_t       block_keys,             // keys each block covers per launch
    int            partial_len,            // hex nibbles for partial match
    uint64_t       jump_size,
    int            deny_hex_len,
    int            save_candidates,
    int            max_candidates,
    FoundResult   *out_result,
    CandidateEntry*out_candidates,
    int           *out_cand_count,
    uint64_t      *out_checked
)
{
    // Shared memory:
    //   sh_dx[256]: deltaX values — 256 × 32 = 8192 bytes
    //   sh_base:    base point (GPoint = 64 bytes)
    //   sh_priv:    current private key (32 bytes)
    //   sh_found:   found flag (4 bytes)
    __shared__ uint256_t sh_dx[256];
    __shared__ GPoint    sh_base;          // use GPoint directly, not separate x/y
    __shared__ uint256_t sh_priv;
    __shared__ int       sh_found;

    int tid = (int)threadIdx.x;

    if (tid == 0) sh_found = 0;
    __syncthreads();

    // Compute this block's starting private key:
    //   block_start = range_start + blockIdx.x * block_keys
    if (tid == 0) {
        typedef unsigned __int128 u128;
        u128 prod = (u128)(uint64_t)blockIdx.x * (u128)block_keys;
        uint256_t offset;
        offset.d[0] = (uint64_t)prod;
        offset.d[1] = (uint64_t)(prod >> 64);
        offset.d[2] = 0; offset.d[3] = 0;

        u256_add_device(&sh_priv, &range_start, &offset);

        // Compute base = sh_priv * G using GTable (up to 32 point additions)
        gpt_compute_pubkey(&sh_base, d_gtbl, &sh_priv);
    }
    __syncthreads();

    uint64_t local_checked = 0ULL;

    for (int batch = 0; batch < BATCHES_PER_BLOCK; batch++) {

        if (sh_found) break;

        // Check range
        if (tid == 0 && u256_cmp(&sh_priv, &range_end) > 0) {
            sh_found = -1; // done with range
        }
        __syncthreads();
        if (sh_found) break;
        // ---- Step 1: compute deltaX[tid] = plus[tid].x - base.x ----
        // Use fe_sub to handle modular arithmetic
        fe_sub(&sh_dx[tid], &d_pl[tid].x, &sh_base.x);
        __syncthreads();

        // ---- Step 2: each thread computes its own inverse (Fermat) ----
        uint256_t inv_dx;
        // deltaX[0] = 0 - base.x = -base.x (plus[0]=infinity has x=0), non-zero, OK to invert
        fe_inv(&inv_dx, &sh_dx[tid]);

        // ---- Step 3: compute 2 points and hash ----
        // Cache base coordinates in registers
        uint256_t bx = sh_base.x, by = sh_base.y;

        // Process FORWARD point: base + plus[tid]  (skip tid==0: plus[0]=infinity)
        if (tid != 0) {
            GPoint fwd;
            gpt_add_with_inv(&fwd,
                             &bx, &by,
                             &d_pl[tid].x, &d_pl[tid].y,
                             &inv_dx);

            // Serialize to compressed pubkey
            uint8_t pubkey[33];
            pubkey[0] = (fwd.y.d[0] & 1) ? 0x03 : 0x02;
            for (int i = 0; i < 4; i++) {
                uint64_t w = fwd.x.d[3-i];
                for (int j = 0; j < 8; j++)
                    pubkey[1 + i*8 + j] = (uint8_t)(w >> (56 - j*8));
            }

            // Public-key deny check
            bool denied = false;
            if (deny_hex_len > 0) {
                int full = deny_hex_len / 2;
                bool ok = true;
                for (int k = 0; k < full && ok; k++)
                    if (pubkey[1+k] != 0x00) ok = false;
                if (ok && (deny_hex_len & 1))
                    if ((pubkey[1+full] & 0xF0) != 0x00) ok = false;
                denied = ok;
            }

            if (!denied) {
                uint8_t h160[20];
                hash160(pubkey, 33, h160);
                local_checked++;

                // Full match check
                bool full_match = true;
                for (int k = 0; k < 20 && full_match; k++)
                    full_match = (h160[k] == target_h160[k]);
                if (full_match) {
                    if (atomicCAS(&out_result->flag, 0, 1) == 0) {
                        // private key = sh_priv + tid
                        u256_add_small(&out_result->priv_key, &sh_priv, (uint32_t)tid);
                        memcpy(out_result->pubkey,  pubkey, 33);
                        memcpy(out_result->hash160, h160, 20);
                    }
                    sh_found = 1;
                }

                // Partial match check
                if (!full_match && partial_len > 0 && !sh_found) {
                    int pb = partial_len / 2;
                    bool pmatch = true;
                    for (int k = 0; k < pb && pmatch; k++)
                        pmatch = (h160[k] == target_h160[k]);
                    if (pmatch && (partial_len & 1))
                        pmatch = ((h160[pb] & 0xF0) == (target_h160[pb] & 0xF0));
                    if (pmatch && save_candidates) {
                        int idx = atomicAdd(out_cand_count, 1);
                        if (idx < max_candidates) {
                            u256_add_small(&out_candidates[idx].priv_key,
                                           &sh_priv, (uint32_t)tid);
                            memcpy(out_candidates[idx].pubkey,  pubkey, 33);
                            memcpy(out_candidates[idx].hash160, h160, 20);
                        }
                    }
                }
            } else {
                local_checked++;
            }
        } // tid != 0 forward

        // Process BACKWARD point: base + minus[tid]  (skip tid==0: minus[0]=infinity)
        if (tid != 0) {
            GPoint bwd;
            // minus[tid].x == plus[tid].x, minus[tid].y = -plus[tid].y
            // deltaX is the same (plus[tid].x - base.x), so same inv_dx
            gpt_add_with_inv(&bwd,
                             &bx, &by,
                             &d_mn[tid].x, &d_mn[tid].y,
                             &inv_dx);

            uint8_t pubkey[33];
            pubkey[0] = (bwd.y.d[0] & 1) ? 0x03 : 0x02;
            for (int i = 0; i < 4; i++) {
                uint64_t w = bwd.x.d[3-i];
                for (int j = 0; j < 8; j++)
                    pubkey[1 + i*8 + j] = (uint8_t)(w >> (56 - j*8));
            }

            bool denied = false;
            if (deny_hex_len > 0) {
                int full = deny_hex_len / 2;
                bool ok = true;
                for (int k = 0; k < full && ok; k++)
                    if (pubkey[1+k] != 0x00) ok = false;
                if (ok && (deny_hex_len & 1))
                    if ((pubkey[1+full] & 0xF0) != 0x00) ok = false;
                denied = ok;
            }

            if (!denied) {
                uint8_t h160[20];
                hash160(pubkey, 33, h160);
                local_checked++;

                bool full_match = true;
                for (int k = 0; k < 20 && full_match; k++)
                    full_match = (h160[k] == target_h160[k]);
                if (full_match) {
                    if (atomicCAS(&out_result->flag, 0, 1) == 0) {
                        // private key = sh_priv - tid
                        uint256_t step_val; u256_set64(&step_val, (uint64_t)tid);
                        u256_sub(&out_result->priv_key, &sh_priv, &step_val);
                        memcpy(out_result->pubkey,  pubkey, 33);
                        memcpy(out_result->hash160, h160, 20);
                    }
                    sh_found = 1;
                }

                if (!full_match && partial_len > 0 && !sh_found) {
                    int pb = partial_len / 2;
                    bool pmatch = true;
                    for (int k = 0; k < pb && pmatch; k++)
                        pmatch = (h160[k] == target_h160[k]);
                    if (pmatch && (partial_len & 1))
                        pmatch = ((h160[pb] & 0xF0) == (target_h160[pb] & 0xF0));
                    if (pmatch && save_candidates) {
                        int idx = atomicAdd(out_cand_count, 1);
                        if (idx < max_candidates) {
                            uint256_t sv; u256_set64(&sv, (uint64_t)tid);
                            u256_sub(&out_candidates[idx].priv_key, &sh_priv, &sv);
                            memcpy(out_candidates[idx].pubkey,  pubkey, 33);
                            memcpy(out_candidates[idx].hash160, h160, 20);
                        }
                    }
                }
            } else {
                local_checked++;
            }
        } // tid != 0 backward

        // Also process priv itself (the base point = priv * G)
        if (tid == 0) {
            uint8_t pubkey[33];
            pubkey[0] = (sh_base.y.d[0] & 1) ? 0x03 : 0x02;
            for (int i = 0; i < 4; i++) {
                uint64_t w = sh_base.x.d[3-i];
                for (int j = 0; j < 8; j++)
                    pubkey[1 + i*8 + j] = (uint8_t)(w >> (56 - j*8));
            }
            uint8_t h160[20];
            hash160(pubkey, 33, h160);
            local_checked++;

            bool full_match = true;
            for (int k = 0; k < 20 && full_match; k++)
                full_match = (h160[k] == target_h160[k]);
            if (full_match && atomicCAS(&out_result->flag, 0, 1) == 0) {
                out_result->priv_key = sh_priv;
                memcpy(out_result->pubkey, pubkey, 33);
                memcpy(out_result->hash160, h160, 20);
                sh_found = 1;
            }
            if (!full_match && partial_len > 0) {
                int pb = partial_len / 2;
                bool pmatch = true;
                for (int k = 0; k < pb && pmatch; k++)
                    pmatch = (h160[k] == target_h160[k]);
                if (pmatch && (partial_len & 1))
                    pmatch = ((h160[pb] & 0xF0) == (target_h160[pb] & 0xF0));
                if (pmatch && save_candidates) {
                    int idx = atomicAdd(out_cand_count, 1);
                    if (idx < max_candidates) {
                        out_candidates[idx].priv_key = sh_priv;
                        memcpy(out_candidates[idx].pubkey, pubkey, 33);
                        memcpy(out_candidates[idx].hash160, h160, 20);
                    }
                }
            }
        }

        // ---- Step 4: advance base += step510G and priv += 510 (thread 0) ----
        if (tid == 0) {
            gpt_add_direct(&sh_base, &sh_base, d_step);
            u256_add_small(&sh_priv, &sh_priv, 510u);
        }
        __syncthreads();

    } // batch loop

    // Accumulate checked count
    atomicAdd(out_checked, local_checked);
}

// ============================================================
// Host utility functions
// ============================================================

static std::string bytesToHex(const uint8_t *d, size_t n) {
    static const char *lut = "0123456789abcdef";
    std::string s; s.reserve(n*2);
    for (size_t i=0;i<n;i++){
        s.push_back(lut[d[i]>>4]);
        s.push_back(lut[d[i]&0xF]);
    }
    return s;
}

static std::string u256ToHex(const uint256_t &v) {
    char buf[65];
    snprintf(buf, sizeof(buf),
             "%016llx%016llx%016llx%016llx",
             (unsigned long long)v.d[3],
             (unsigned long long)v.d[2],
             (unsigned long long)v.d[1],
             (unsigned long long)v.d[0]);
    return std::string(buf);
}

static void hexToU256(const std::string &hex, uint256_t *out) {
    memset(out, 0, sizeof(uint256_t));
    int len = (int)hex.size();
    for (int i = 0; i < len && i < 64; i++) {
        char c = hex[len - 1 - i];
        uint64_t d = (c>='0'&&c<='9') ? c-'0' :
                     (c>='a'&&c<='f') ? c-'a'+10 :
                     (c>='A'&&c<='F') ? c-'A'+10 : 0;
        out->d[i>>4] |= d << ((i&15)*4);
    }
}

// Simple host-side uint256 add for advancing range
static void h_u256_add_u64(uint256_t *r, const uint256_t *a, uint64_t b) {
    typedef unsigned __int128 u128;
    u128 acc = (u128)a->d[0] + b;
    r->d[0] = (uint64_t)acc; uint64_t c = (uint64_t)(acc>>64);
    acc = (u128)a->d[1] + c; r->d[1]=(uint64_t)acc; c=(uint64_t)(acc>>64);
    acc = (u128)a->d[2] + c; r->d[2]=(uint64_t)acc; c=(uint64_t)(acc>>64);
    r->d[3] = a->d[3] + c;
}

static int h_u256_gt(const uint256_t *a, const uint256_t *b) {
    return h_u256_cmp(a, b) > 0;
}

static std::string formatTime(double sec) {
    int h=(int)sec/3600, m=((int)sec%3600)/60, s=(int)sec%60;
    char buf[32];
    snprintf(buf,sizeof(buf),"%02d:%02d:%02d",h,m,s);
    return buf;
}

static void printStats(const std::string &address,
                       const std::string &hashHex,
                       const std::string &gpuName,
                       double mkeys,
                       uint64_t checked,
                       double elapsed,
                       const std::string &range,
                       long double progress,
                       int saves,
                       uint64_t candidates,
                       uint64_t jumps,
                       bool showCand, bool showJump)
{
    int lines = 10 + (showCand?1:0) + (showJump?1:0);
    static bool first = true;
    if (!first) printf("\033[%dA", lines);
    else        first = false;

    printf("================= WORK IN PROGRESS =================\n");
    printf("Target Address: %s\n", address.c_str());
    printf("Hash160       : %s\n", hashHex.c_str());
    printf("GPU Device    : %s\n", gpuName.c_str());
    printf("Mkeys/s       : %.2f\n", mkeys);
    printf("Total Checked : %llu\n", (unsigned long long)checked);
    printf("Elapsed Time  : %s\n", formatTime(elapsed).c_str());
    printf("Range         : %s\n", range.c_str());
    printf("Progress      : %.4Lf %%\n", progress);
    printf("Progress Save : %d\n", saves);
    if (showCand) printf("Candidates    : %llu\n", (unsigned long long)candidates);
    if (showJump) printf("Jumps         : %llu\n", (unsigned long long)jumps);
    fflush(stdout);
}

static void saveProgress(const std::string &msg) {
    std::ofstream ofs("progress.txt", std::ios::app);
    if (ofs) ofs << msg << "\n";
}

static void writeFoundKey(const std::string &priv, const std::string &pub,
                          const std::string &wif,  const std::string &addr) {
    std::ofstream ofs("found_keys.txt", std::ios::app);
    if (ofs) ofs << priv << " " << pub << " " << wif << " " << addr << "\n";
}

static void saveCandidateToFile(const std::string &priv, const std::string &pub,
                                 const std::string &h160) {
    std::ofstream ofs("candidates.txt", std::ios::app);
    if (ofs) ofs << priv << " " << pub << " " << h160 << "\n";
}

static void printUsage(const char *prog) {
    fprintf(stderr,
        "Usage: %s -a <Base58_P2PKH> -r <START:END>"
        " [-p <HEXLEN>] [-j <JUMP>] [-s] [-t <GPU_ID>]"
        " [--public-deny <HEXLEN>]\n", prog);
}

// ============================================================
// main()
// ============================================================

int main(int argc, char **argv)
{
    // Parse arguments
    std::string targetAddress, rangeStr;
    bool aOK=false, rOK=false;
    int  partialLen  = 0;
    uint64_t jumpSize = 0;
    bool saveCands   = false;
    int  gpuId       = 0;
    int  denyHexLen  = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"-a") && i+1<argc) {
            targetAddress = argv[++i]; aOK = true;
        } else if (!strcmp(argv[i],"-r") && i+1<argc) {
            rangeStr = argv[++i]; rOK = true;
        } else if (!strcmp(argv[i],"-p") && i+1<argc) {
            partialLen = atoi(argv[++i]);
            if (partialLen < 1 || partialLen > 40) {
                fprintf(stderr,"-p must be 1-40\n"); return 1;
            }
        } else if (!strcmp(argv[i],"-j") && i+1<argc) {
            jumpSize = strtoull(argv[++i], nullptr, 10);
        } else if (!strcmp(argv[i],"-s")) {
            saveCands = true;
        } else if (!strcmp(argv[i],"-t") && i+1<argc) {
            gpuId = atoi(argv[++i]);
        } else if (!strcmp(argv[i],"--public-deny") && i+1<argc) {
            denyHexLen = atoi(argv[++i]);
            if (denyHexLen < 1 || denyHexLen > 64) {
                fprintf(stderr,"--public-deny must be 1-64\n"); return 1;
            }
        } else if (!strcmp(argv[i],"-h") || !strcmp(argv[i],"--help")) {
            printUsage(argv[0]); return 0;
        } else {
            printUsage(argv[0]); return 1;
        }
    }
    if (!aOK || !rOK) { printUsage(argv[0]); return 1; }

    // Decode target address
    std::vector<uint8_t> targetHash160;
    try {
        targetHash160 = P2PKHDecoder::getHash160(targetAddress);
    } catch (const std::exception &e) {
        fprintf(stderr,"Address error: %s\n", e.what()); return 1;
    }

    std::string targetHashHex = bytesToHex(targetHash160.data(), 20);

    // Parse range
    size_t colon = rangeStr.find(':');
    if (colon == std::string::npos) { fprintf(stderr,"Bad range\n"); return 1; }
    std::string startHex = rangeStr.substr(0, colon);
    std::string endHex   = rangeStr.substr(colon+1);

    uint256_t rangeStart, rangeEnd;
    hexToU256(startHex, &rangeStart);
    hexToU256(endHex,   &rangeEnd);

    if (h_u256_cmp(&rangeStart, &rangeEnd) > 0) {
        fprintf(stderr,"Range start > end\n"); return 1;
    }

    // Select GPU
    int nDev = 0;
    cudaGetDeviceCount(&nDev);
    if (nDev == 0) { fprintf(stderr,"No CUDA devices found\n"); return 1; }
    if (gpuId >= nDev) { fprintf(stderr,"GPU %d not found\n", gpuId); return 1; }
    cudaSetDevice(gpuId);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpuId);
    std::string gpuName = prop.name;

    printf("=== Cyclone GPU — Optimized Batch Solver ===\n");
    printf("GPU Device    : %s\n", gpuName.c_str());
    printf("Target Address: %s\n", targetAddress.c_str());
    printf("Hash160       : %s\n", targetHashHex.c_str());
    printf("Range         : %s:%s\n", startHex.c_str(), endHex.c_str());
    if (partialLen)  printf("Partial match : %d hex nibbles\n", partialLen);
    if (jumpSize)    printf("Jump size     : %llu\n", (unsigned long long)jumpSize);
    if (saveCands)   printf("Save candidates: yes\n");
    if (denyHexLen)  printf("Public deny   : %d hex nibbles\n", denyHexLen);
    printf("\n");

    // ---- Host precomputation ----
    printf("Precomputing GTable (8192 points) ...\n");
    auto tPrecomp = std::chrono::high_resolution_clock::now();

    std::vector<GPoint> h_gtable(GTABLE_SIZE);
    h_build_gtable(h_gtable.data());

    std::vector<GPoint> h_plus(256), h_minus(256);
    h_build_plus_minus(h_plus.data(), h_minus.data());

    // step510G = 510 * G (used to advance base within kernel)
    uint256_t scalar510;
    h_u256_set64(&scalar510, 510ULL);
    GPoint h_step510G;
    h_gpt_compute_pubkey(&h_step510G, h_gtable.data(), &scalar510);

    auto tPrecompDone = std::chrono::high_resolution_clock::now();
    double precompSec = std::chrono::duration<double>(tPrecompDone - tPrecomp).count();
    printf("GTable precomputed in %.2f seconds.\n\n", precompSec);

    // ---- Allocate and copy device memory ----
    size_t gtableBytes = GTABLE_SIZE * sizeof(GPoint);
    size_t pmBytes     = 256 * sizeof(GPoint);

    cudaMalloc(&d_gtable,   gtableBytes);
    cudaMalloc(&d_plus,     pmBytes);
    cudaMalloc(&d_minus,    pmBytes);
    cudaMalloc(&d_step510G, sizeof(GPoint));

    cudaMemcpy(d_gtable,   h_gtable.data(), gtableBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_plus,     h_plus.data(),   pmBytes,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_minus,    h_minus.data(),  pmBytes,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_step510G, &h_step510G,     sizeof(GPoint), cudaMemcpyHostToDevice);

    // Target hash160 on device
    uint8_t *d_target;
    cudaMalloc(&d_target, 20);
    cudaMemcpy(d_target, targetHash160.data(), 20, cudaMemcpyHostToDevice);

    // Result, candidates, counters
    const int MAX_CANDS = 65536;
    cudaMalloc(&d_result,     sizeof(FoundResult));
    cudaMalloc(&d_candidates, MAX_CANDS * sizeof(CandidateEntry));
    cudaMalloc(&d_cand_count, sizeof(int));
    cudaMalloc(&d_checked,    sizeof(uint64_t));

    // Initialize
    FoundResult initResult; initResult.flag = 0;
    cudaMemcpy(d_result, &initResult, sizeof(FoundResult), cudaMemcpyHostToDevice);
    int initCand = 0;
    cudaMemcpy(d_cand_count, &initCand, sizeof(int), cudaMemcpyHostToDevice);
    uint64_t initChecked = 0ULL;
    cudaMemcpy(d_checked, &initChecked, sizeof(uint64_t), cudaMemcpyHostToDevice);

    // ---- Kernel launch parameters ----
    // Grid: use enough blocks to fill the GPU
    int numSMs = prop.multiProcessorCount;
    // Aim for ~8 blocks per SM for good occupancy
    int gridBlocks = numSMs * 8;
    if (gridBlocks < 64)  gridBlocks = 64;
    if (gridBlocks > 8192) gridBlocks = 8192;

    // Keys covered per block per kernel launch
    uint64_t block_keys = (uint64_t)BATCHES_PER_BLOCK * 510ULL;
    // Keys per kernel launch
    uint64_t launch_keys = (uint64_t)gridBlocks * block_keys;

    printf("Grid blocks   : %d\n", gridBlocks);
    printf("Batches/block : %d\n", BATCHES_PER_BLOCK);
    printf("Keys/launch   : ~%llu M\n\n",
           (unsigned long long)(launch_keys / 1000000));

    // ---- Main search loop ----
    auto tStart   = std::chrono::high_resolution_clock::now();
    auto lastStat = tStart;
    auto lastSave = tStart;

    uint64_t totalChecked  = 0ULL;
    uint64_t totalCands    = 0ULL;
    int      progressSaves = 0;
    bool     found         = false;

    // Compute approximate total range as long double for progress %
    // Simple estimate: range = 2^(bits in endHex) - 2^(bits in startHex)
    long double totalRange = 0.0L;
    {
        // Approximate: count bits of rangeEnd
        int bits = 0;
        for (int i = 3; i >= 0; i--) {
            if (rangeEnd.d[i]) {
                bits = i * 64;
                uint64_t v = rangeEnd.d[i];
                while (v) { bits++; v >>= 1; }
                break;
            }
        }
        totalRange = powl(2.0L, (long double)bits);
    }

    std::string displayRange = startHex + ":" + endHex;

    uint256_t curStart = rangeStart;

    while (!found && h_u256_cmp(&curStart, &rangeEnd) <= 0) {

        // Check for early exit if result was found in previous launch
        {
            FoundResult hr;
            cudaMemcpy(&hr, d_result, sizeof(FoundResult), cudaMemcpyDeviceToHost);
            if (hr.flag) { found = true; break; }
        }

        // Reset checked counter for this launch
        initChecked = 0ULL;
        cudaMemcpy(d_checked, &initChecked, sizeof(uint64_t), cudaMemcpyHostToDevice);

        // Launch kernel
        batch_search_kernel<<<gridBlocks, BLOCK_THREADS>>>(
            d_gtable, d_plus, d_minus, d_step510G,
            d_target,
            curStart, rangeEnd,
            block_keys,
            partialLen, jumpSize, denyHexLen,
            saveCands ? 1 : 0, MAX_CANDS,
            d_result, d_candidates, d_cand_count, d_checked
        );

        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr,"CUDA error: %s\n", cudaGetErrorString(err));
            break;
        }

        // Read back checked count
        uint64_t launchChecked = 0;
        cudaMemcpy(&launchChecked, d_checked, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        totalChecked += launchChecked;

        // Read back candidates
        {
            int ncands = 0;
            cudaMemcpy(&ncands, d_cand_count, sizeof(int), cudaMemcpyDeviceToHost);
            if (ncands > 0) {
                int take = std::min(ncands, MAX_CANDS);
                std::vector<CandidateEntry> cands(take);
                cudaMemcpy(cands.data(), d_candidates,
                           take * sizeof(CandidateEntry),
                           cudaMemcpyDeviceToHost);
                for (int ci = 0; ci < take; ci++) {
                    std::string privHex = u256ToHex(cands[ci].priv_key);
                    std::string pubHex  = bytesToHex(cands[ci].pubkey, 33);
                    std::string h160Hex = bytesToHex(cands[ci].hash160, 20);
                    totalCands++;
                    if (saveCands) saveCandidateToFile(privHex, pubHex, h160Hex);
                }
                // Reset candidates
                initCand = 0;
                cudaMemcpy(d_cand_count, &initCand, sizeof(int), cudaMemcpyHostToDevice);
            }
        }

        // Check for full match
        {
            FoundResult hr;
            cudaMemcpy(&hr, d_result, sizeof(FoundResult), cudaMemcpyDeviceToHost);
            if (hr.flag) { found = true; break; }
        }

        // Advance start
        h_u256_add_u64(&curStart, &curStart, launch_keys);

        // Status display
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - tStart).count();
        if (std::chrono::duration<double>(now - lastStat).count() >= 5.0) {
            double mkeys = (elapsed > 0) ? (double)totalChecked / elapsed / 1e6 : 0.0;
            long double prog = (totalRange > 0) ?
                (long double)totalChecked / totalRange * 100.0L : 0.0L;
            printStats(targetAddress, targetHashHex, gpuName,
                       mkeys, totalChecked, elapsed, displayRange,
                       prog, progressSaves, totalCands, 0ULL,
                       saveCands, false);
            lastStat = now;
        }

        // Periodic progress save
        if (std::chrono::duration<double>(now - lastSave).count() >= 300.0) {
            progressSaves++;
            double elapsed2 = std::chrono::duration<double>(now - tStart).count();
            double mkeys2 = (elapsed2 > 0) ? (double)totalChecked/elapsed2/1e6 : 0.0;
            std::ostringstream oss;
            oss << "Progress Save #" << progressSaves
                << " at " << elapsed2 << " sec:"
                << " TotalChecked=" << totalChecked
                << " Mkeys/s=" << std::fixed << std::setprecision(2) << mkeys2
                << " CurrentKey=" << u256ToHex(curStart);
            saveProgress(oss.str());
            lastSave = now;
        }
    }

    // Final status
    auto tEnd = std::chrono::high_resolution_clock::now();
    double totalElapsed = std::chrono::duration<double>(tEnd - tStart).count();
    double finalMkeys   = (totalElapsed > 0) ? (double)totalChecked/totalElapsed/1e6 : 0.0;

    if (found) {
        FoundResult hr;
        cudaMemcpy(&hr, d_result, sizeof(FoundResult), cudaMemcpyDeviceToHost);

        std::string privHex = u256ToHex(hr.priv_key);
        std::string pubHex  = bytesToHex(hr.pubkey, 33);
        std::string h160Hex = bytesToHex(hr.hash160, 20);
        std::string wif;
        try { wif = P2PKHDecoder::compute_wif(privHex, true); }
        catch (...) { wif = "(wif error)"; }

        writeFoundKey(privHex, pubHex, wif, targetAddress);

        printf("\n================== FOUND MATCH! ==================\n");
        printf("Private Key   : %s\n", privHex.c_str());
        printf("Public Key    : %s\n", pubHex.c_str());
        printf("WIF           : %s\n", wif.c_str());
        printf("Hash160       : %s\n", h160Hex.c_str());
        printf("P2PKH Address : %s\n", targetAddress.c_str());
    } else {
        printf("\nNo match found in range.\n");
    }

    printf("\nTotal Checked : %llu\n", (unsigned long long)totalChecked);
    printf("Elapsed Time  : %s\n", formatTime(totalElapsed).c_str());
    printf("Speed         : %.2f Mkeys/s\n", finalMkeys);
    if (totalCands) printf("Candidates    : %llu\n", (unsigned long long)totalCands);

    // Cleanup
    cudaFree(d_gtable);   cudaFree(d_plus);   cudaFree(d_minus);
    cudaFree(d_step510G); cudaFree(d_target);
    cudaFree(d_result);   cudaFree(d_candidates);
    cudaFree(d_cand_count); cudaFree(d_checked);

    return found ? 0 : 1;
}
