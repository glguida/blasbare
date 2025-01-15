#include "stdint.h"
#include "stdio.h"
#include "pthread.h"

#include <nux/nux.h>
#include <nux/plt.h>

uint64_t ggml_cycles (void)
{
  printf ("GGML CYCLES!\n");
  return hal_cpu_cycles ();
}

uint64_t ggml_time_us (void)
{
  printf ("GGML TIMER US!\n");
  /* XXX: MICROSECONDS! */
  return plt_tmr_ctr ();
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
  return 0;
}

int pthread_join(pthread_t thread, void *retval)
{
  printf ("GGML PTHREAD JOIN!\n");
  return 0;
}

void sched_yield (void)
{
  printf ("GGML SCHED YIELD!\n");
}
