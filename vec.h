#ifndef VEC_H
#define VEC_H

typedef struct {
    float *data;
    int    len;
} Vec;

#define MAX_VEC_SZ 2048

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
