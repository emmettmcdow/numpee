#ifndef VEC_H
#define VEC_H

typedef struct {
    float *data;
    int    len;
} Vec;

// Lifecycle
Vec*  vec_create(int len);
void  vec_free(Vec *v);

// Element-wise
Vec*  vec_add(Vec *a, Vec *b);
Vec*  vec_sub(Vec *a, Vec *b);
Vec*  vec_mul(Vec *a, Vec *b);
Vec*  vec_div(Vec *a, Vec *b);
Vec*  vec_scale(Vec *a, float s);

// Reductions
float vec_sum(Vec *a);
float vec_dot(Vec *a, Vec *b);
float vec_min(Vec *a);
float vec_max(Vec *a);

// Utilities
void  vec_print(Vec *a);

#endif
