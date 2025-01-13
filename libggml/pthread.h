typedef void *pthread_t;

int pthread_create(pthread_t * out, void * unused, void *(*func)(void *), void * arg);
int pthread_join(pthread_t thread, void * unused);

