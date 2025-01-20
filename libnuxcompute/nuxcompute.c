#include <stdio.h>
#include <nux/locks.h>
#include <nux/nux.h>
#include <nux/cpumask.h>
#include <stree.h>
#include "nuxcompute.h"

#define NUXCOMPUTE_DEBUG
#ifndef NUXCOMPUTE_DEBUG
#define NCPRINT(...)
#else
extern lock_t printlock;
#define NCPRINT(...) ({ spinlock (&printlock); printf(__VA_ARGS__); spinunlock(&printlock); })
#endif

lock_t nc_lock;
static WORD_T *nc_cpus;
static struct nc_fnarg {
  void (*func)(void *arg);
  void *arg;
} nc_exe[HAL_MAXCPUS];
static cpumask_t nc_owned_cpus;

/*
  This is accessed atomically.
*/
static cpumask_t nc_running_cpumask;

/*
  NUX Compute CPU allocator.

  There's a pool of CPUs that are assigned to NUX compute. CPUs in this pool
  can be allocated to run a specific function.
*/

static bool
_nc_cpu_owned (unsigned cpu)
{
  assert (cpu < HAL_MAXCPUS);
  return cpumask_get (&nc_owned_cpus, cpu);
}

unsigned
nuxcompute_allocate_cpu (void (*fn)(void *arg), void *arg)
{
  const unsigned nc_cpus_order = stree_order(HAL_MAXCPUS);
  long cpu;

  spinlock(&nc_lock);
  cpu = stree_bitsearch (nc_cpus, nc_cpus_order, 1);
  if (cpu >= 0)
    stree_clrbit (nc_cpus, nc_cpus_order, cpu);
  assert (_nc_cpu_owned (cpu));
  nc_exe[cpu].func = fn;
  nc_exe[cpu].arg = arg;

  spinunlock (&nc_lock);

  cpu_ipi (cpu);

  NCPRINT ("Sallocated cpu %ld\n", cpu);
  return cpu < 0 ? CPU_INVALID : (unsigned)cpu;
}

void
nuxcompute_free_cpu (unsigned cpu)
{
  const unsigned nc_cpus_order = stree_order(HAL_MAXCPUS);

  spinlock(&nc_lock);
  assert (_nc_cpu_owned (cpu));
  nc_exe[cpu].func = NULL;
  nc_exe[cpu].arg = NULL;
  stree_setbit (nc_cpus, nc_cpus_order, cpu);
  spinunlock (&nc_lock);

  NCPRINT ("free cpu %d\n", cpu);
}

void
nuxcompute_add_cpu (unsigned cpu)
{
  const unsigned nc_cpus_order = stree_order(HAL_MAXCPUS);
  assert (cpu < HAL_MAXCPUS);

  spinlock(&nc_lock);
  nc_exe[cpu].func = NULL;
  nc_exe[cpu].arg = NULL;
  stree_setbit (nc_cpus, nc_cpus_order, cpu);
  cpumask_set (&nc_owned_cpus, cpu);
  spinunlock (&nc_lock);
  NCPRINT ("add cpu %d\n", cpu);
}

void
nuxcompute_wait_cpu (unsigned cpu)
{
  assert (cpu < HAL_MAXCPUS);

  NCPRINT ("waiting cpu %d...\n", cpu);

  while (!atomic_cpumask_get (&nc_running_cpumask, cpu))
    hal_cpu_relax();

  NCPRINT ("done\n");
}

void
nuxcompute_cpu_run (void)
{
  unsigned cpu = cpu_id ();
  struct nc_fnarg fnarg;

  spinlock (&nc_lock);
  assert (_nc_cpu_owned (cpu));
  fnarg = nc_exe[cpu];
  spinunlock (&nc_lock);

  atomic_cpumask_set (&nc_running_cpumask, cpu);

  printf ("CPU RUN %d [%p(%p)]\n", cpu, fnarg.func, fnarg.arg);
  if (fnarg.func != NULL)
    fnarg.func (fnarg.arg);

  atomic_cpumask_clear (&nc_running_cpumask, cpu);
}

bool
nuxcompute_cpu_owned (unsigned cpu)
{
  bool ret;

  spinlock (&nc_lock);
  ret = _nc_cpu_owned (cpu);
  spinunlock (&nc_lock);

  return ret;
}

void
nuxcompute_init (void)
{
  vaddr_t buf;
  const unsigned nc_cpus_order = stree_order(HAL_MAXCPUS);

  printf ("NUXCOMPUTE INIT!\n");

  buf = kmem_alloc (0, STREE_SIZE(nc_cpus_order));
  assert (buf != VADDR_INVALID);
  nc_cpus = (void *)buf;
}
