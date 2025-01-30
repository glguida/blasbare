#include <nux/nux.h>
#include <nux/nuxperf.h>
#include <assert.h>
#include "stdlib.h"

#define NUXPERF_DECLARE
#include "perf.h"

#define MAGIC 0x66001DA

struct malloc_header {
  unsigned long magic;
  unsigned long size;
};

const char *nux_symresolve(unsigned long);

void *malloc (size_t size)
{
  vaddr_t va;
  struct malloc_header *ptr;

  nuxperf_inc(&ggmlux_malloc);
  nuxmeasure_add(&ggmlux_allocated, size);

  va = kmem_alloc(0, size + sizeof(struct malloc_header));
  ptr = (struct malloc_header *)va;


  //printf("{A %d} [%s]", size, nux_symresolve(__builtin_return_address(0)));

  ptr->magic = MAGIC;
  ptr->size = size;

  return (void *)(ptr+1);
}

void free (void *buf)
{
  if (buf == NULL)
    return;

  struct malloc_header *ptr = (struct malloc_header *)buf - 1;
  if (ptr->magic != MAGIC)
    {
      printf("UNMATCHED MAGIC on PTR %p (buf %p) [%lx != %lx]\n",
	     ptr, buf, ptr->magic, MAGIC);
      return;
    }

  nuxperf_inc(&ggmlux_free);
  nuxmeasure_add(&ggmlux_freed, ptr->size);
  //printf("{F %d} [%s]", ptr->size, nux_symresolve(__builtin_return_address(0)));

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
  if (buf == NULL)
    return malloc(size);

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
