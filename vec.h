#ifndef VEC_H
#define VEC_H

typedef struct {
    float *data;
    int    len;
} Vec;

#define MAX_VEC_SZ 2048

#if defined(__x86_64__) || defined(_M_X64) || defined(__x86_64__) || defined(_M_X64)
// x86
#define PLAT_PRE(f) _x86_simd_ ## f
#define __X86__
#else
// Fallback
#define PLAT_PRE(f) _cpu_ ## f
#define __GENERIC_ARCH__
#endif

// Lifecycle
Vec*  vec_create(int len);
void  vec_destroy(Vec *v);

// Element-wise
Vec* vec_add(Vec *a, Vec *b, Vec *out);
Vec* vec_sub(Vec *a, Vec *b, Vec *out);
Vec* vec_mul(Vec *a, Vec *b, Vec *out);
Vec* vec_div(Vec *a, Vec *b, Vec *out);
Vec* vec_scale(Vec *a, float s, Vec *out);

// Reductions
float vec_sum(Vec *a);
float vec_dot(Vec *a, Vec *b);
float vec_min(Vec *a);
float vec_max(Vec *a);

// Utilities
void  vec_print(Vec *a);
void vec_zero(Vec *a);
void vec_ones(Vec *a);

#endif

// Helpers
#define MIN(a, b) ((a) < (b) ? (a) : (b))
