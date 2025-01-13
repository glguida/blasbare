#include_next <stdio.h>

typedef void FILE;

#define stdout ((FILE *)1)
#define stderr ((FILE *)2)

/* For now, always print in stdout */
#define fprintf(_f, _a, ...) printf(_a, ##__VA_ARGS__)
#define vfprintf(_f, _a, _b) vprintf(_a, _b)
