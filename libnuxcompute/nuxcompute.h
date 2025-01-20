#ifndef _NUXCOMPUTE_H
#define _NUXCOMPUTE_H

#define CPU_INVALID ((unsigned)-1)

unsigned nuxcompute_allocate_cpu (void (*fn)(void *arg), void *arg);
void nuxcompute_free_cpu (unsigned cpu);
void nuxcompute_add_cpu (unsigned cpu);
void nuxcompute_wait_cpu (unsigned cpu);

bool nuxcompute_cpu_owned (unsigned cpu);

bool nuxcompute_start (void (*init)(void *), void *arg);
void nuxcompute_stop (void);

void nuxcompute_cpu_run (void);
void nuxcompute_init (void);

#endif /* _NUXCOMPUTE_H */
