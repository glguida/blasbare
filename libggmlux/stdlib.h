#include <stddef.h>

#include_next <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

  void *malloc(size_t size);
  void free (void *ptr);
  void *calloc (size_t, size_t);
  void *realloc (void *, size_t);

  int posix_memalign (void **memptr, size_t aln, size_t size);

  void qsort(void *a, size_t n, size_t es,
	     int (*cmp)(const void *, const void *));

#ifdef __cplusplus
}
#endif
