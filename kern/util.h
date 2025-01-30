#ifndef _UTIL_H
#define _UTIL_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <queue.h>
#include <stdbool.h>
#include <assert.h>

/*
  Trivial hash map implementation.
*/

struct hash_e {
  void *keyptr;
  size_t keylen;
  void *val;
  LIST_ENTRY(hash_e) hashq;
};

#define HASH_SIZE 255 /* Prime */

struct hashmap {
  LIST_HEAD(, hash_e) map[HASH_SIZE];
};
void hashmap_init(struct hashmap *hm);
void hashmap_add(struct hashmap *hm, char *keyptr, size_t keylen, void *val);
bool hashmap_get(struct hashmap *hm, void *key, size_t keylen, void **valout);


struct vocab_e {
  void *ptr;
  size_t len;
  TAILQ_ENTRY(vocab_e) hashq;
};

#define VOCAB_MAXHASH (1 << 16)

struct vocab {
  int32_t max_idx;
  struct vocab_e *vocab;
  TAILQ_HEAD(, vocab_e) revocab[VOCAB_MAXHASH];
};

void vocab_init(struct vocab *v, int32_t maxidx);
void vocab_add(struct vocab *v, int32_t idx, unsigned len, char *ptr);
bool vocab_find(struct vocab *v, char *s, int32_t *idout);
int32_t vocab_maxidx(struct vocab *v);
void vocab_print(struct vocab *v, int32_t idx);

struct mapped_file {
  const char *buffer;
  size_t size;
  size_t pos;
};

void mfile_init(struct mapped_file *mf, const char *buffer, size_t size);
const char *mfile_curptr(struct mapped_file *mf);
void mfile_skip(struct mapped_file *mf, size_t bytes);
void mfile_read(struct mapped_file *mf, char *out, size_t bytes);
bool mfile_eof(struct mapped_file *mf);

char * asprintf (const char *fmt, ...);


struct fvec {
  float *buf;
  size_t cap;
  size_t count;
};

static inline void fvec_init(struct fvec *v)
{
  v->buf = NULL;
  v->cap = 0;
  v->count = 0;
}

static inline void fvec_ensure(struct fvec *v, size_t count)
{
  if (v->cap >= count)
    return;

  v->buf = realloc(v->buf, count * sizeof(float));
  v->cap = count;
}

static inline void fvec_copy_array(struct fvec *v, float *a, size_t count)
{
  fvec_ensure (v, count);
  assert (v->cap >= count);
  memcpy(v->buf, a, count * sizeof(float));
  v->count = count;
}

static inline size_t fvec_size(struct fvec *v)
{
  return v->count;
}

static inline void fvec_free(struct fvec *v)
{
  if (v->buf)
    {
      assert(v->cap != 0);
      free(v->buf);
      v->buf = NULL;
      v->cap = 0;
      v->count = 0;
    }
  assert (v->buf == NULL);
  assert (v->cap == 0);
  assert (v->count == 0);
}

static inline void fvec_pushback (struct fvec *v, float f)
{
  if (v->count == v->cap)
    {
      v->buf = realloc(v->buf, v->cap + 32 * sizeof(float));
      v->cap += 32;
    }
  assert (v->count < v->cap);
  v->buf[v->count++] = f;
}

static inline float *fvec_data (struct fvec *v)
{
  return v->buf;
}


struct idvec {
  int32_t *buf;
  size_t cap;
  size_t count;
};

static inline void idvec_init(struct idvec *v)
{
  v->buf = NULL;
  v->cap = 0;
  v->count = 0;
}

static inline void idvec_ensure(struct idvec *v, size_t count)
{
  if (v->cap >= count)
    return;

  v->buf = realloc(v->buf, count * sizeof(int32_t));
  v->cap = count;
}

static inline void idvec_copy_array(struct idvec *v, int32_t *a, size_t count)
{
  idvec_ensure (v, count);
  assert (v->cap >= count);
  memcpy(v->buf, a, count * sizeof(int32_t));
  v->count = count;
}

static inline size_t idvec_size(struct idvec *v)
{
  return v->count;
}

static inline void idvec_free(struct idvec *v)
{
  if (v->buf)
    {
      assert(v->cap != 0);
      free(v->buf);
      v->buf = NULL;
      v->cap = 0;
      v->count = 0;
    }
  assert (v->buf == NULL);
  assert (v->cap == 0);
  assert (v->count == 0);
}

static inline void idvec_pushback (struct idvec *v, int32_t f)
{
  if (v->count == v->cap)
    {
      v->buf = realloc(v->buf, v->cap + 32 * sizeof(int32_t));
      v->cap += 32;
    }
  assert (v->count < v->cap);
  v->buf[v->count++] = f;
}

static inline int32_t *idvec_data (struct idvec *v)
{
  return v->buf;
}


#define PRIf "(%ld/10000)"
#define FLOATPRINT(_f) ((long)((_f)*10000))

#endif
