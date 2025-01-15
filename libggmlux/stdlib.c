#include <nux/nux.h>
#include <assert.h>
#include "stdlib.h"

#define MAGIC 0x7001DA

struct malloc_header {
  unsigned long magic;
  unsigned long size;
};

void *malloc (size_t size)
{
  vaddr_t va;
  struct malloc_header *ptr;

  va = kmem_alloc(0, size + sizeof(struct malloc_header));
  ptr = (struct malloc_header *)va;

  ptr->magic = MAGIC;
  ptr->size = size;

  return (void *)(ptr+1);
}

void free (void *buf)
{
  struct malloc_header *ptr = (struct malloc_header *)buf - 1;

  assert (ptr->magic == MAGIC);
  kmem_free (0, (vaddr_t)ptr, ptr->size + sizeof (struct malloc_header));
}

void *calloc (size_t nmemb, size_t size)
{
  void *ptr;

  ptr = malloc (nmemb * size);
  memset (ptr, 0, nmemb * size);
  return ptr;
}

void *realloc (void *buf, size_t size)
{
  struct malloc_header *ptr = (struct malloc_header *)buf - 1;

  assert (ptr->magic == MAGIC);
  
  /*
    Just reallocate a buffer and copy the content.
  */
  void *buf2 = malloc (size);

  memcpy (buf2, buf, size < ptr->size ? size : ptr->size);

  free(buf);

  return buf2;
}

int posix_memalign (void **memptr, size_t alignment, size_t size)
{
  (void)alignment; /* Ignore alignment, for now. */
  *memptr = malloc (size);
  return 0;
}
