#include "util.h"
#include <stdlib.h>
#include <string.h>

static inline uint16_t
_hashfn(char *ptr, size_t len)
{
  uint16_t hash = 0;

  for (int i = 0; i < len; i++)
    hash += ptr[i];

  return hash % HASH_SIZE;
}

void
hashmap_init(struct hashmap *hm)
{
  for (int i = 0; i < HASH_SIZE; i++)
    LIST_INIT(hm->map + i);
}

void
hashmap_add(struct hashmap *hm, char *keyptr, size_t keylen, void *val)
{
  struct hash_e *e;
  uint16_t hash = _hashfn(keyptr, keylen);

  e = malloc(sizeof(struct hash_e));
  e->keyptr = keyptr;
  e->keylen = keylen;
  e->val = val;

  LIST_INSERT_HEAD(hm->map + hash, e, hashq);
}

bool
hashmap_get(struct hashmap *hm, void *key, size_t keylen, void **valout)
{
  struct hash_e *e;
  uint16_t hash = _hashfn(key, keylen);

  LIST_FOREACH(e, hm->map + hash, hashq)
    {
      if ((e->keylen == keylen) && !memcmp(e->keyptr, key, keylen))
	{
	  *valout = e->val;
	  return true;
	}
    }
  return false;
}


void
vocab_init(struct vocab *v, int32_t maxidx)
{
  v->vocab = malloc(sizeof(struct vocab_e) * maxidx);
  v->max_idx = maxidx;

  for (int i = 0; i < VOCAB_MAXHASH; i++)
    TAILQ_INIT(v->revocab + i);
}

static uint16_t
vocab_hash(char *ptr, size_t len)
{
  uint16_t hash = 0;

  for (int i = 0; i < len; i++)
    hash += ptr[i];

  return hash;
}

void
vocab_add(struct vocab *v, int32_t idx, unsigned len, char *ptr)
{
  uint32_t hash;
  
  assert (idx < v->max_idx);
  v->vocab[idx].ptr = ptr;
  v->vocab[idx].len = len;



  hash = vocab_hash(ptr, len);

  TAILQ_INSERT_TAIL(v->revocab + hash, v->vocab + idx, hashq);
}

bool
vocab_find(struct vocab *v, char *s, int32_t *idout)
{
  struct vocab_e *e;
  size_t slen = strlen(s);
  uint16_t hash = vocab_hash(s, slen);

  TAILQ_FOREACH(e, v->revocab + hash, hashq)
    {
      if ((e->len == slen) && !memcmp(e->ptr, s, slen))
	{
	  *idout = ((uintptr_t)e - (uintptr_t)v->vocab)/sizeof(struct vocab_e);
	  return true;
	}
    }
  return false;
}

int32_t
vocab_maxidx(struct vocab *v)
{
  return v->max_idx;
}

void
vocab_print(struct vocab *v, int32_t idx)
{
  assert (idx < v->max_idx);
  size_t len = v->vocab[idx].len;
  char *string = __builtin_alloca(len + 1);
  memcpy(string, v->vocab[idx].ptr, len);
  string[len] = '\0';
  printf("%s", string);
}

void
mfile_init(struct mapped_file *mf, const char *buffer, size_t size)
{
  mf->buffer = buffer;
  mf->size = size;
  mf->pos = 0;
}

const char *
mfile_curptr(struct mapped_file *mf)
{
  return mf->buffer + mf->pos;
}

void
mfile_skip(struct mapped_file *mf, size_t bytes)
{
  mf->pos += bytes;
}

void
mfile_read(struct mapped_file *mf, char *out, size_t bytes)
{
  memcpy(out, mf->buffer + mf->pos, bytes);
  mf->pos += bytes;
}

bool
mfile_eof(struct mapped_file *mf)
{
  return mf->pos >= mf->size;
}

char *
asprintf(const char *fmt, ...)
{
  char *ptr;
  va_list ap;
  char buf[256];

  va_start (ap, fmt);
  vsnprintf(buf, 256, fmt, ap);
  va_end (ap);

  size_t len = strlen(buf);
  assert(len < 256);

  ptr = (char *)malloc(len + 1);
  memcpy (ptr, buf, len);
  ptr[len] = '\0';
  return ptr;
}
