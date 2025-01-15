#ifndef _GGMLUX_STRING_H
#define _GGMLUX_STRING_H

#include_next <string.h>

#ifdef __cplusplus
extern "C" {
#endif

  char *strdup (const char *s);

  int strcmp (const char *s1, const char *s2);

#ifdef __cplusplus
}
#endif

#endif
