#include <stdlib.h>
#include <stdio.h>
#include "vec.h"
#include <assert.h>
#include <immintrin.h>

// ***************************************************************************** Universal CPU SISD
#ifdef NUMPEE_GENERIC
static float* _cpu_vec_add(float *a, float *b, float *out, int len) {
  for (float *p = a, *q = b, *r = out; p < a + len;) {
      *(r++) = *(p++) + *(q++);
  }
  return out;
}

static float* _cpu_vec_sub(float *a, float *b, float *out, int len) {
  for (float *p = a, *q = b, *r = out; p < a + len;) {
      *(r++) = *(p++) - *(q++);
  }
  return out;
}

static float* _cpu_vec_mul(float *a, float *b, float *out, int len) {
  for (float *p = a, *q = b, *r = out; p < a + len;) {
      *(r++) = *(p++) * *(q++);
  }
  return out;
}

static float* _cpu_vec_div(float *a, float *b, float *out, int len) {
  for (float *p = a, *q = b, *r = out; p < a + len;) {
      *(r++) = *(p++) / *(q++);
  }
  return out;
}

static float* _cpu_vec_scale(float *a, float s, float *out, int len) {
  for (float *p = a, *q = out; p < a + len;) {
      *(q++) = *(p++) * s;
  }
  return out;
}

static float _cpu_vec_sum(float *a, int len) {
  float output = 0;
  for (float *p=a; p < a + len; p++) output += *p;
  return output;
}

static float _cpu_vec_dot(float *a, float *b, int len) {
  float buf[MAX_VEC_SZ];
  _cpu_vec_mul(a, b, buf, len);
  return _cpu_vec_sum(buf, len);
}

static float _cpu_vec_min(float *a, int len) {
  float min = a[0];
  for (float *p=a + 1; p < a + len; p++) {
    if (*p < min) min = *p;
  }
  return min;
}

static float _cpu_vec_max(float *a, int len) {
  float max = a[0];
  for (float *p=a + 1; p < a + len; p++) {
    if (*p > max) max = *p;
  }
  return max;
}
#endif


// *********************************************************************************** x86 CPU SIMD
#ifdef NUMPEE_X86
#include <immintrin.h>

static float* _x86_simd_vec_add(float *a, float *b, float *out, int len) {
  int i = 0;
  for (; i <= len - 8; i += 8) {
    __m256 a_vec = _mm256_loadu_ps(&a[i]);    // Load
    __m256 b_vec = _mm256_loadu_ps(&b[i]);
    __m256 sum = _mm256_add_ps(a_vec, b_vec); // Add
    _mm256_storeu_ps(out + i, sum);           // Store
  }
  for (; i < len; i++) {out[i] = a[i] + b[i];}
  return out;
}

static float* _x86_simd_vec_sub(float *a, float *b, float *out, int len) {
  int i = 0;
  for (; i <= len - 8; i += 8) {
      __m256 a_vec = _mm256_loadu_ps(&a[i]);    // Load
      __m256 b_vec = _mm256_loadu_ps(&b[i]);
      __m256 dif = _mm256_sub_ps(a_vec, b_vec); // Sub
      _mm256_storeu_ps(out + i, dif);           // Store
  }
  for (; i < len; i++) {out[i] = a[i] - b[i];}
  return out;
}

static float* _x86_simd_vec_mul(float *a, float *b, float *out, int len) {
  int i = 0;
  for (; i <= len - 8; i += 8) {
    __m256 a_vec = _mm256_loadu_ps(&a[i]);     // Load
    __m256 b_vec = _mm256_loadu_ps(&b[i]);
    __m256 prod = _mm256_mul_ps(a_vec, b_vec); // Mul
    _mm256_storeu_ps(out + i, prod);           // Store
  }
  for (; i < len; i++) {out[i] = a[i] * b[i];}
  return out;
}

static float* _x86_simd_vec_div(float *a, float *b, float *out, int len) {
  int i = 0;
  for (; i <= len - 8; i += 8) {
    __m256 a_vec = _mm256_loadu_ps(&a[i]);    // Load
    __m256 b_vec = _mm256_loadu_ps(&b[i]);
    __m256 sum = _mm256_div_ps(a_vec, b_vec); // Div
    _mm256_storeu_ps(out + i, sum);           // Store
  }
  for (; i < len; i++) {out[i] = a[i] / b[i];}
  return out;
}

static float* _x86_simd_vec_scale(float *a, float s, float *out, int len) {
  int i = 0;
  float scale_v[8] = {s, s, s, s, s, s, s, s};
  __m256 b_vec = _mm256_loadu_ps(scale_v);
  for (; i <= len - 8; i += 8) {
    __m256 a_vec = _mm256_loadu_ps(a + i);    // Load
    __m256 prod = _mm256_mul_ps(a_vec, b_vec); // Mul
    _mm256_storeu_ps(out + i, prod);           // Store
  }
  for (; i < len; i++) {out[i] = a[i] * s;}
  return out;
}

static float _x86_simd_vec_sum(float *a, int len) {
  int i = 0;
  float tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (; i < MIN(len, 8); i++) tmp[i] = a[i];
  __m256 tmp_vec = _mm256_loadu_ps(tmp);
  for (; i <= len - 8; i += 8) {
    __m256 a_vec = _mm256_loadu_ps(a + i);    // Load
    tmp_vec = _mm256_add_ps(a_vec, tmp_vec);  // Add
  }
  _mm256_storeu_ps(tmp, tmp_vec);             // Store
  float output = 0;
  for (; i < len; i++) {output += a[i];}
  for (i = 0; i < 8; i++) {output += tmp[i];}
  return output;
}

static float _x86_simd_vec_dot(float *a, float *b, int len) {
  float buf[MAX_VEC_SZ];
  _x86_simd_vec_mul(a, b, buf, len);
  return _x86_simd_vec_sum(buf, len);
}

static float _x86_simd_vec_min(float *a, int len) {
  float min = a[0];
  for (float *p=a + 1; p < a + len; p++) {
    if (*p < min) min = *p;
  }
  return min;
}

static float _x86_simd_vec_max(float *a, int len) {
  float max = a[0];
  for (float *p=a + 1; p < a + len; p++) {
    if (*p > max) max = *p;
  }
  return max;
}
#endif


Vec* vec_create(int len) {
  Vec *output = malloc(sizeof(Vec));
  if (output == NULL) return NULL;
  float *buf = malloc(sizeof(float) * len);
  if (output == NULL) return NULL;
  output->data = buf;
  output->len = len;
  return output;
}

void vec_destroy(Vec *v) {
  free(v->data);
  free(v);
}

Vec* vec_add(Vec *a, Vec *b, Vec *out) {
  assert(a->len == b->len && a->len == out->len);
  PLAT_PRE(vec_add)(a->data, b->data, out->data, a->len);
  return out;
}

Vec* vec_sub(Vec *a, Vec *b, Vec *out) {
  assert(a->len == b->len && a->len == out->len);
  PLAT_PRE(vec_sub)(a->data, b->data, out->data, a->len);
  return out;
}

Vec* vec_mul(Vec *a, Vec *b, Vec *out) {
  assert(a->len == b->len && a->len == out->len);
  PLAT_PRE(vec_mul)(a->data, b->data, out->data, a->len);
  return out;
}

Vec* vec_div(Vec *a, Vec *b, Vec *out) {
  assert(a->len == b->len && a->len == out->len);
  PLAT_PRE(vec_div)(a->data, b->data, out->data, a->len);
  return out;
}

Vec* vec_scale(Vec *a, float s, Vec *out) {
  assert(a->len == out->len);
  PLAT_PRE(vec_scale)(a->data, s, out->data, a->len);
  return out;
}

float vec_sum(Vec *a) {
  return PLAT_PRE(vec_sum)(a->data, a->len);
}

float vec_dot(Vec *a, Vec *b) {
  assert(a->len == b->len);
  assert(a->len < MAX_VEC_SZ);
  return PLAT_PRE(vec_dot)(a->data, b->data, a->len);
}

float vec_min(Vec *a) {
  return PLAT_PRE(vec_min)(a->data, a->len);
}

float vec_max(Vec *a) {
  return PLAT_PRE(vec_max)(a->data, a->len);
}

void  vec_print(Vec *a) {
  printf("[");
  for (float *p=a->data; p < a->data + a->len; p++) {printf("%f,", *p);}
  printf("]\n");
}
void vec_zero(Vec *a) {for (float *p=a->data; p < a->data + a->len; p++) *p = 0;}
void vec_ones(Vec *a) {for (float *p=a->data; p < a->data + a->len; p++) *p = 1;}

#ifdef TEST

static void test_create_zero(void) {
  printf("%s...", __func__);
  Vec *v = vec_create(3);
  assert(v->len == 3);
  v->data[0] = 1.0;
  v->data[1] = 1.0;
  v->data[2] = 1.0;
  vec_zero(v);
  for (float *p = v->data; p < v->data + v->len; p++) assert(*p== 0);
  printf(" PASS\n");
}

static void test_ones_scale(void) {
  printf("%s...", __func__);
  Vec *v = vec_create(9);
  vec_ones(v);
  Vec *v2 = vec_create(9);
  vec_scale(v, 10, v2);
  for (float *p = v2->data; p < v2->data + v2->len; p++) assert(*p== 10);
  printf(" PASS\n");
}

static void test_mdas(void) {
  printf("%s...", __func__);
  Vec *v = vec_create(9);
  vec_ones(v);
  vec_add(v, v, v);
  for (float *p = v->data; p < v->data + v->len; p++) assert(*p== 2);
  vec_mul(v, v, v);
  for (float *p = v->data; p < v->data + v->len; p++) assert(*p== 4);

  Vec *v2 = vec_create(9);
  vec_ones(v2);
  vec_scale(v2, 2, v2);

  vec_div(v, v2, v);
  for (float *p = v->data; p < v->data + v->len; p++) assert(*p== 2);
  vec_sub(v, v2, v);
  for (float *p = v->data; p < v->data + v->len; p++) assert(*p== 0);
  printf(" PASS\n");
}

static void test_sum(void) {
  printf("%s...", __func__);
  float buf[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Vec v = {.data = buf, .len = 9};
  assert(vec_sum(&v) == 45);

  printf(" PASS\n");
}

static void test_dot(void) {
  printf("%s...", __func__);
  Vec *v = vec_create(3);
  v->data[0] = 1;
  v->data[1] = 2;
  v->data[2] = 3;

  Vec *v2 = vec_create(3);
  v2->data[0] = 4;
  v2->data[1] = 5;
  v2->data[2] = 6;

  assert(vec_dot(v, v2) == 32);
  printf(" PASS\n");
}

static void test_minmax(void) {
  printf("%s...", __func__);
  Vec *v = vec_create(3);
  v->data[2] = 1;
  v->data[1] = 2;
  v->data[0] = 3;

  assert(vec_min(v) == 1);
  assert(vec_max(v) == 3);
  printf(" PASS\n");
}

static void test_weighted_sq_diff(void) {
  printf("%s...", __func__);
  int len = 1000;
  Vec *a    = vec_create(len);
  Vec *b    = vec_create(len);
  Vec *diff = vec_create(len);
  Vec *sq   = vec_create(len);
  Vec *w    = vec_create(len);

  for (int i = 0; i < len; i++) { a->data[i] = 3.0f; b->data[i] = 1.0f; }
  vec_sub(a, b, diff);        // all 2.0
  vec_mul(diff, diff, sq);    // all 4.0
  vec_scale(sq, 0.5f, w);     // all 2.0
  float result = vec_sum(w);  // 2000.0

  assert(result == 2000.0f);

  vec_destroy(a); vec_destroy(b); vec_destroy(diff);
  vec_destroy(sq); vec_destroy(w);
  printf(" PASS\n");
}

static void test_dot_identity(void) {
  printf("%s...", __func__);
  int len = 1000;
  Vec *a   = vec_create(len);
  Vec *sq  = vec_create(len);

  for (int i = 0; i < len; i++) a->data[i] = (float)i;

  float via_dot = vec_dot(a, a);
  vec_mul(a, a, sq);
  float via_sum = vec_sum(sq);

  assert(via_dot == via_sum);

  vec_destroy(a); vec_destroy(sq);
  printf(" PASS\n");
}

int main(void) {
  test_create_zero();
  test_ones_scale();
  test_mdas();
  test_sum();
  test_dot();
  test_minmax();
  test_weighted_sq_diff();
  test_dot_identity();
  return 0;
}

#endif

#ifdef BENCHMARK
#include <time.h>

static void test_dot_identity(int len) {
  Vec *a   = vec_create(len);
  Vec *sq  = vec_create(len);

  for (int i = 0; i < len; i++) a->data[i] = (float)i;

  float via_dot = vec_dot(a, a);
  vec_mul(a, a, sq);
  float via_sum = vec_sum(sq);

  assert(via_dot == via_sum);

  vec_destroy(a); vec_destroy(sq);
}

int main(void) {
#if defined(NUMPEE_X86)
  printf("Benchmarking x86\n");
#elif defined(NUMPEE_GENERIC)
  printf("Benchmarking generic\n");
#endif
  time_t start = time(NULL);
  int N_ITERATIONS = 1000;
  for (int n = 0; n < N_ITERATIONS; n++) {
    for (int i = 1; i < MAX_VEC_SZ; i++) {
      test_dot_identity(i);
    }
  }
  time_t end = time(NULL);

  printf("Completed benchmark in %ld seconds\n", end - start);
  return 0;
}
#endif
