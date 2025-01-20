/*
  NUX: A kernel Library.
  Copyright (C) 2019 Gianluca Guida, glguida@tlbflush.org

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2
  of the License, or (at your option) any later version.

  See COPYING file for the full license.

  SPDX-License-Identifier:	GPL2.0+
*/

#include <assert.h>
#include <stdio.h>
#include <nux/nux.h>
#include <nux/hal.h>
#include <nuxcompute.h>

lock_t printlock;
#define printf(...) ({ spinlock (&printlock); printf(__VA_ARGS__); spinunlock(&printlock); })

uctxt_t u_init;
struct hal_umap umap;

int nuxcompute_initialized;


extern void _simple_init (void *arg);

int
main (int argc, char *argv[])
{
  nuxcompute_init ();
  cpu_ipi (cpu_id ());

  for (unsigned i = 0; i < cpu_num(); i++)
    if (i != cpu_id())
      nuxcompute_add_cpu (i);

  __atomic_store_n (&nuxcompute_initialized, 1, __ATOMIC_SEQ_CST);

  nuxcompute_allocate_cpu (_simple_init, NULL);
  
  printf("DONE");
  return EXIT_IDLE;
}

int
main_ap (void)
{
  while (!__atomic_load_n (&nuxcompute_initialized, __ATOMIC_SEQ_CST));

  return EXIT_IDLE;
}

uctxt_t *
entry_sysc (uctxt_t * u,
	    unsigned long a1, unsigned long a2, unsigned long a3,
	    unsigned long a4, unsigned long a5, unsigned long a6,
	    unsigned long a7)
{
  fatal ("Received unknown syscall %ld %ld %ld %ld %ld %ld %ld\n",
	 a1, a2, a3, a4, a5, a6, a7);
  return u;
}

uctxt_t *
entry_ipi (uctxt_t * uctxt)
{
  if (nuxcompute_cpu_owned (cpu_id()))
    {
      printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
      nuxcompute_cpu_run ();
    }

  return uctxt;
}

uctxt_t *
entry_alarm (uctxt_t * uctxt)
{
  timer_alarm (1 * 1000 * 1000 * 1000);
  info ("TMR: %" PRIu64 " us", timer_gettime ());
  uctxt_print (uctxt);
  return uctxt;
}

uctxt_t *
entry_ex (uctxt_t * uctxt, unsigned ex)
{
  info ("Exception %d", ex);
  uctxt_print (uctxt);
  return UCTXT_IDLE;
}

uctxt_t *
entry_pf (uctxt_t * uctxt, vaddr_t va, hal_pfinfo_t pfi)
{
  info ("CPU #%d Pagefault at %08lx (%d)", cpu_id (), va, pfi);
  uctxt_print (uctxt);
  return UCTXT_IDLE;
}

uctxt_t *
entry_irq (uctxt_t * uctxt, unsigned irq, bool lvl)
{
  info ("IRQ %d", irq);
  return uctxt;

}
