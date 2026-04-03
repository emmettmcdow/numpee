#include <stdlib.h>
#include <stdio.h>
#include "vec.h"
#include <assert.h>
#include <immintrin.h>

// *********************************************************************************** Universal CPU SISD
float* _cpu_vec_add(float *a, float *b, float *out, size_t len) {
  for (float *p = a, *q = b, *r = out; p < a + len;) {
      *(r++) = *(p++) + *(q++);
  }
  return out;
}

float* _cpu_vec_sub(float *a, float *b, float *out, size_t len) {
  for (float *p = a, *q = b, *r = out; p < a + len;) {
      *(r++) = *(p++) - *(q++);
  }
  return out;
}

float* _cpu_vec_mul(float *a, float *b, float *out, size_t len) {
  for (float *p = a, *q = b, *r = out; p < a + len;) {
      *(r++) = *(p++) * *(q++);
  }
  return out;
}

float* _cpu_vec_div(float *a, float *b, float *out, size_t len) {
  for (float *p = a, *q = b, *r = out; p < a + len;) {
      *(r++) = *(p++) / *(q++);
  }
  return out;
}

float* _cpu_vec_scale(float *a, float s, float *out, size_t len) {
  for (float *p = a, *q = out; p < a + len;) {
      *(q++) = *(p++) * s;
  }
  return out;
}

float _cpu_vec_sum(float *a, size_t len) {
  float output = 0;
  for (float *p=a; p < a + len; p++) output += *p;  
  return output;
}

float _cpu_vec_dot(float *a, float *b, size_t len) {
  float buf[MAX_VEC_SZ];
  _cpu_vec_mul(a, b, buf, len);
  return _cpu_vec_sum(buf, len);
}

float _cpu_vec_min(float *a, size_t len) {
  float min = a[0];
  for (float *p=a + 1; p < a + len; p++) {
    if (*p < min) min = *p;
  }
  return min;
}

float _cpu_vec_max(float *a, size_t len) {
  float max = a[0];
  for (float *p=a + 1; p < a + len; p++) {
    if (*p > max) max = *p;
  }
  return max;
}


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
  _cpu_vec_add(a->data, b->data, out->data, a->len);
  return out;
}

Vec* vec_sub(Vec *a, Vec *b, Vec *out) {
  assert(a->len == b->len && a->len == out->len);
  _cpu_vec_sub(a->data, b->data, out->data, a->len);
  return out;
}

Vec* vec_mul(Vec *a, Vec *b, Vec *out) {
  assert(a->len == b->len && a->len == out->len);
  _cpu_vec_mul(a->data, b->data, out->data, a->len);
  return out;
}

Vec* vec_div(Vec *a, Vec *b, Vec *out) {
  assert(a->len == b->len && a->len == out->len);
  _cpu_vec_div(a->data, b->data, out->data, a->len);
  return out;
}

Vec* vec_scale(Vec *a, float s, Vec *out) {
  assert(a->len == out->len);
  _cpu_vec_scale(a->data, s, out->data, a->len);
  return out;
}

float vec_sum(Vec *a) {
  return _cpu_vec_sum(a->data, a->len);
}

float vec_dot(Vec *a, Vec *b) {
  assert(a->len == b->len);
  assert(a->len < MAX_VEC_SZ);
  return _cpu_vec_dot(a->data, b->data, a->len);
}

float vec_min(Vec *a) {
  return _cpu_vec_min(a->data, a->len);
}

float vec_max(Vec *a) {
  return _cpu_vec_max(a->data, a->len);
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
  Vec *v = vec_create(3);
  vec_ones(v);
  Vec *v2 = vec_create(3);
  vec_scale(v, 10, v2);
  for (float *p = v2->data; p < v2->data + v2->len; p++) assert(*p== 10);
  printf(" PASS\n");
}

static void test_mdas(void) {
  printf("%s...", __func__);
  Vec *v = vec_create(3);
  vec_ones(v);
  vec_add(v, v, v);
  for (float *p = v->data; p < v->data + v->len; p++) assert(*p== 2);
  vec_mul(v, v, v);
  for (float *p = v->data; p < v->data + v->len; p++) assert(*p== 4);

  Vec *v2 = vec_create(3);
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
  Vec *v = vec_create(3);
  v->data[0] = 1;
  v->data[1] = 2;
  v->data[2] = 3;
  assert(vec_sum(v) == 6);

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

int main(void) {
  test_create_zero();
  test_ones_scale();
  test_mdas();
  test_sum();
  test_dot();
  test_minmax();
  return 0;
}

#endif
