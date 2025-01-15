/*
  As we compile with built-ins, some printfs are changed to 'puts'.

  Change them back. :-)
*/

#include <stdio.h>

int puts (const char *s)
{
  printf("%s", s);
}
