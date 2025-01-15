#include_next <stdio.h>

typedef void FILE;

#define stdout ((FILE *)1)
#define stderr ((FILE *)2)

#ifdef __cplusplus
extern "C" {
#endif

/* For now, always print in stdout */
#define fprintf(_f, _a, ...) printf(_a, ##__VA_ARGS__)
#define vfprintf(_f, _a, _b) vprintf(_a, _b)
#define fputs(_t, _s) printf("%s", _t)

#define fflush(_f)

#ifdef __cplusplus
}
#endif
