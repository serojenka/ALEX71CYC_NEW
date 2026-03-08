#ifndef GPU_SECP256K1_CUH
#define GPU_SECP256K1_CUH

#include "gpu_uint256.cuh"

// Affine EC point. {x=0, y=0} represents the point at infinity.
typedef struct { uint256_t x; uint256_t y; } GPoint;

// secp256k1 generator G
__device__ __constant__ uint256_t GPU_GX = {{
    0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL
}};
__device__ __constant__ uint256_t GPU_GY = {{
    0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL
}};

// ============================================================
// Device: affine point-at-infinity test
// ============================================================
__device__ __forceinline__ int gpt_is_inf(const GPoint *p) {
    return u256_is_zero(&p->x) && u256_is_zero(&p->y);
}

// ============================================================
// Device: point addition (both affine, non-equal, non-infinity)
// Computes r = p1 + p2 using modular inverse of (p2.x - p1.x)
// Safe when r aliases p1 or p2 — saves inputs before writing.
// ============================================================
__device__ void gpt_add_direct(GPoint *r,
                                const GPoint *p1,
                                const GPoint *p2)
{
    // Handle point at infinity
    if (gpt_is_inf(p1)) { *r = *p2; return; }
    if (gpt_is_inf(p2)) { *r = *p1; return; }

    // Save input coordinates before any writes to r (handles r == p1 or r == p2)
    uint256_t p1x = p1->x, p1y = p1->y;
    uint256_t p2x = p2->x, p2y = p2->y;

    uint256_t dx, dy, lam, lam2, tmp;
    fe_sub(&dx, &p2x, &p1x);
    fe_sub(&dy, &p2y, &p1y);

    if (u256_is_zero(&dx)) {
        // Same x: points are equal (doubling) or inverses (infinity)
        if (u256_is_zero(&dy)) {
            // Point doubling: lambda = 3*x^2 / (2*y)
            fe_sqr(&tmp, &p1x);
            fe_add(&lam, &tmp, &tmp); fe_add(&lam, &lam, &tmp); // 3x^2
            fe_add(&dy, &p1y, &p1y); // 2y
            fe_inv(&tmp, &dy);
            fe_mul(&lam, &lam, &tmp);
        } else {
            // Inverse points → infinity
            u256_set_zero(&r->x); u256_set_zero(&r->y);
            return;
        }
    } else {
        fe_inv(&tmp, &dx);
        fe_mul(&lam, &dy, &tmp);
    }

    // rx = lambda^2 - p1.x - p2.x
    fe_sqr(&lam2, &lam);
    fe_sub(&r->x, &lam2, &p1x);
    fe_sub(&r->x, &r->x, &p2x);
    // ry = lambda*(p1.x - rx) - p1.y
    fe_sub(&tmp, &p1x, &r->x);
    fe_mul(&r->y, &lam, &tmp);
    fe_sub(&r->y, &r->y, &p1y);
}

// ============================================================
// Device: point addition using PRECOMPUTED inverse of (p2.x - base.x)
// Used in the batch loop — no Fermat inversion needed here.
//   r = base + p2  where inv_dx = 1/(p2.x - base.x)
// ============================================================
__device__ __forceinline__ void gpt_add_with_inv(
    GPoint *r,
    const uint256_t *bx, const uint256_t *by,
    const uint256_t *px, const uint256_t *py,
    const uint256_t *inv_dx)
{
    uint256_t dy, lam, lam2, tmp;
    fe_sub(&dy, py, by);
    fe_mul(&lam, &dy, inv_dx);   // lambda = (p.y - base.y) / (p.x - base.x)
    fe_sqr(&lam2, &lam);
    fe_sub(&r->x, &lam2, bx);
    fe_sub(&r->x, &r->x, px);   // rx = lam^2 - base.x - p.x
    fe_sub(&tmp, bx, &r->x);
    fe_mul(&r->y, &lam, &tmp);
    fe_sub(&r->y, &r->y, by);   // ry = lam*(base.x - rx) - base.y
}

// ============================================================
// Device: compute public key from 256-bit scalar using GTable
// GTable layout: GTable[i*256 + j] = (j+1) * 256^i * G  (j=0..254)
//                GTable[i*256 + 255] = 256^(i+1) * G  (not used in lookup)
// GetByte(i) = byte i (i=0 = LSB byte of scalar)
// ============================================================
__device__ void gpt_compute_pubkey(GPoint *result,
                                    const GPoint *gtable,
                                    const uint256_t *scalar)
{
    GPoint Q; u256_set_zero(&Q.x); u256_set_zero(&Q.y); // Q = infinity

    for (int i = 0; i < 32; i++) {
        // Extract byte i (i=0 = LSB)
        uint8_t b = (uint8_t)((scalar->d[i >> 3] >> ((i & 7) << 3)) & 0xFF);
        if (b == 0) continue;

        const GPoint *entry = &gtable[i * 256 + (b - 1)];
        if (gpt_is_inf(&Q)) {
            Q = *entry;
        } else {
            gpt_add_direct(&Q, &Q, entry);
        }
    }
    *result = Q;
}

// ============================================================
// Host versions (for CPU-side GTable + plus/minus precomputation)
// ============================================================

static inline int h_gpt_is_inf(const GPoint *p) {
    return h_u256_is_zero(&p->x) && h_u256_is_zero(&p->y);
}

static inline void h_gpt_add_direct(GPoint *r,
                                     const GPoint *p1,
                                     const GPoint *p2)
{
    if (h_gpt_is_inf(p1)) { *r = *p2; return; }
    if (h_gpt_is_inf(p2)) { *r = *p1; return; }

    // Save input coordinates before any writes to r (handles r == p1 or r == p2)
    uint256_t p1x = p1->x, p1y = p1->y;
    uint256_t p2x = p2->x, p2y = p2->y;

    uint256_t dx, dy, lam, lam2, tmp;
    h_fe_sub(&dx, &p2x, &p1x);
    h_fe_sub(&dy, &p2y, &p1y);

    if (h_u256_is_zero(&dx)) {
        if (h_u256_is_zero(&dy)) {
            h_fe_sqr(&tmp, &p1x);
            h_fe_add(&lam, &tmp, &tmp); h_fe_add(&lam, &lam, &tmp);
            h_fe_add(&dy, &p1y, &p1y);
            h_fe_inv(&tmp, &dy);
            h_fe_mul(&lam, &lam, &tmp);
        } else {
            h_u256_set64(&r->x, 0); h_u256_set64(&r->y, 0);
            return;
        }
    } else {
        h_fe_inv(&tmp, &dx);
        h_fe_mul(&lam, &dy, &tmp);
    }
    h_fe_sqr(&lam2, &lam);
    h_fe_sub(&r->x, &lam2, &p1x);
    h_fe_sub(&r->x, &r->x, &p2x);
    h_fe_sub(&tmp, &p1x, &r->x);
    h_fe_mul(&r->y, &lam, &tmp);
    h_fe_sub(&r->y, &r->y, &p1y);
}

// Compute scalar * G using GTable on the host
static inline void h_gpt_compute_pubkey(GPoint *result,
                                         const GPoint *gtable,
                                         const uint256_t *scalar)
{
    GPoint Q; h_u256_set64(&Q.x, 0); h_u256_set64(&Q.y, 0);
    for (int i = 0; i < 32; i++) {
        uint8_t b = (uint8_t)((scalar->d[i >> 3] >> ((i & 7) << 3)) & 0xFF);
        if (!b) continue;
        const GPoint *entry = &gtable[i * 256 + (b - 1)];
        if (h_gpt_is_inf(&Q)) { Q = *entry; }
        else { h_gpt_add_direct(&Q, &Q, entry); }
    }
    *result = Q;
}

// Build the 8192-point GTable on the host (same algorithm as SECP256K1::Init)
static inline void h_build_gtable(GPoint *gtable)
{
    // secp256k1 G
    GPoint G;
    G.x.d[0]=0x59F2815B16F81798ULL; G.x.d[1]=0x029BFCDB2DCE28D9ULL;
    G.x.d[2]=0x55A06295CE870B07ULL; G.x.d[3]=0x79BE667EF9DCBBACULL;
    G.y.d[0]=0x9C47D08FFB10D4B8ULL; G.y.d[1]=0xFD17B448A6855419ULL;
    G.y.d[2]=0x5DA4FBFC0E1108A8ULL; G.y.d[3]=0x483ADA7726A3C465ULL;

    GPoint N = G;
    for (int i = 0; i < 32; i++) {
        gtable[i * 256 + 0] = N;
        GPoint base = N;
        h_gpt_add_direct(&N, &N, &N); // double
        for (int j = 1; j < 255; j++) {
            gtable[i * 256 + j] = N;
            h_gpt_add_direct(&N, &N, &base);
        }
        gtable[i * 256 + 255] = N; // dummy / 256^(i+1)*G
    }
}

// Build plus[i] = i*G (i=0..255) and minus[i] = -(i*G) on the host
// plus[0] = point at infinity (0*G)
static inline void h_build_plus_minus(GPoint *plus, GPoint *minus)
{
    GPoint G;
    G.x.d[0]=0x59F2815B16F81798ULL; G.x.d[1]=0x029BFCDB2DCE28D9ULL;
    G.x.d[2]=0x55A06295CE870B07ULL; G.x.d[3]=0x79BE667EF9DCBBACULL;
    G.y.d[0]=0x9C47D08FFB10D4B8ULL; G.y.d[1]=0xFD17B448A6855419ULL;
    G.y.d[2]=0x5DA4FBFC0E1108A8ULL; G.y.d[3]=0x483ADA7726A3C465ULL;

    // plus[0] = 0*G = infinity
    h_u256_set64(&plus[0].x, 0);
    h_u256_set64(&plus[0].y, 0);
    h_fe_sub(&minus[0].x, &plus[0].x, &plus[0].x); // 0
    h_fe_sub(&minus[0].y, &plus[0].y, &plus[0].y); // 0

    GPoint cur = G;
    for (int i = 1; i < 256; i++) {
        plus[i] = cur;
        minus[i].x = cur.x;
        h_fe_sub(&minus[i].y, &H_P, &cur.y); // negate Y
        if (i < 255) h_gpt_add_direct(&cur, &cur, &G);
    }
}

#endif // GPU_SECP256K1_CUH
