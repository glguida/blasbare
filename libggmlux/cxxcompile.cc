#include "assert.h"
#include "stdlib.h"
#include "string.h"

namespace __cxxabiv1 
{
	/* guard variables */

	/* The ABI requires a 64-bit type.  */
	__extension__ typedef int __guard __attribute__((mode(__DI__)));

	extern "C" int __cxa_guard_acquire (__guard *);
	extern "C" void __cxa_guard_release (__guard *);
	extern "C" void __cxa_guard_abort (__guard *);

	extern "C" int __cxa_guard_acquire (__guard *g) 
	{
		return !*(char *)(g);
	}

	extern "C" void __cxa_guard_release (__guard *g)
	{
		*(char *)g = 1;
	}

	extern "C" void __cxa_guard_abort (__guard *)
	{

	}
}

namespace std {
  void __throw_out_of_range(char const *str) {
    assert(!"out of range");
  }
};



void *operator new(size_t size)
{
  void *ptr = malloc(size);
  return ptr;
}

void *operator new[](size_t size)
{
  void *ptr = malloc(size);
  return ptr;
}

void operator delete(void *p)
{
  free(p);
}

void operator delete(void *p, size_t size)
{
  free(p);
}

void operator delete[](void *p)
{
  free(p);
}

extern "C" {
  void *__dso_handle = NULL;
}
