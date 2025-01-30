#include "stdint.h"
#include "stdio.h"
#include "pthread.h"

#include <nux/nux.h>
#include <nux/plt.h>

#include <nuxcompute.h>

#define printf(...)

uint64_t ggml_cycles (void)
{
  printf ("GGML CYCLES!\n");
  return hal_cpu_cycles ();
}

uint64_t ggml_time_us (void)
{
  printf ("GGML TIMER US!\n");
  return timer_gettime () / 1000;
}

void ggml_time_init (void)
{
  printf ("GGML TIMER INIT!\n");
}

int pthread_create(pthread_t *restrict thread,
		   void *attr,
		   void *(*start_routine)(void *),
		   void *restrict arg)
{
  printf ("GGML PTHREAD CREATE!\n");
  unsigned cpu = nuxcompute_allocate_cpu (start_routine, arg);

  printf("CPU IS %d\n", cpu);

  if (cpu == CPU_INVALID)
    return -1;

  *thread = cpu;
  return 0;
}

int pthread_join(pthread_t thread, void * unused)
{
  unsigned cpu = thread;

  printf ("GGML PTHREAD JOIN!\n");
  nuxcompute_wait_cpu (cpu);
  return 0;
}

void sched_yield (void)
{
  printf ("GGML SCHED YIELD!\n");
}
