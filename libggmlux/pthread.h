#include <nux/locks.h>

typedef void *pthread_t;
typedef struct {
  int waiters;
  int generation;
} pthread_cond_t;
typedef lock_t pthread_mutex_t;

#ifdef __cplusplus
extern "C" {
#endif

  int pthread_create(pthread_t * out, void * unused, void *(*func)(void *), void * arg);
  int pthread_join(pthread_t thread, void * unused);

#ifdef _cplusplus
}
#endif

static inline int pthread_mutex_init (pthread_mutex_t *mutex, void *attr)
{
  (void)attr;
  spinlock_init (mutex);
  return 0;
}

static inline int pthread_mutex_lock (pthread_mutex_t *mutex)
{
  spinlock (mutex);
  return 0;
}

static inline int pthread_mutex_unlock (pthread_mutex_t *mutex)
{
  spinunlock (mutex);
  return 0;
}

static inline int pthread_mutex_destroy (pthread_mutex_t *mutex)
{
  (void)mutex;
  return 0;
}



static inline void pthread_cond_init(pthread_cond_t* cond, void *attr)
{
  (void)attr;
  __atomic_store_n(&cond->waiters, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&cond->generation, 0, __ATOMIC_RELAXED);
}

static inline void pthread_cond_wait(pthread_cond_t* cond, lock_t* mutex)
{
  __atomic_add_fetch(&cond->waiters, 1, __ATOMIC_RELAXED);
  int gen = __atomic_load_n(&cond->generation, __ATOMIC_ACQUIRE);

    spinunlock(mutex);

    while (true) {
      int current_gen = __atomic_load_n(&cond->generation, __ATOMIC_ACQUIRE);
      if (current_gen != gen) {
	break;
      }

      hal_cpu_relax ();
    }
    __atomic_sub_fetch(&cond->waiters, 1, __ATOMIC_RELAXED);
    spinlock(mutex);
}

static inline void pthread_cond_signal(pthread_cond_t* cond)
{
  if (__atomic_load_n(&cond->waiters, __ATOMIC_RELAXED) == 0) {
    return;
  }
  __atomic_add_fetch(&cond->generation, 1, __ATOMIC_RELEASE);
}

static inline void pthread_cond_broadcast(pthread_cond_t* cond)
{
  if (__atomic_load_n(&cond->waiters, __ATOMIC_RELAXED) == 0) {
    return;
  }
  __atomic_add_fetch(&cond->generation, 1, __ATOMIC_RELEASE);
}

static inline void pthread_cond_destroy(pthread_cond_t *cond)
{
  (void)cond;
}
