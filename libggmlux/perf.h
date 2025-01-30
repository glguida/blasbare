#include <nux/nuxperf.h>

#ifdef NUXPERF_DECLARE
#define NUXPERF(_s) extern nuxperf_t __perf _s
#define NUXMEASURE(_s) extern nuxmeasure_t __measure _s
#endif

#ifdef NUXPERF_DEFINE
#define NUXPERF(_s) nuxperf_t __perf _s = { .name = #_s , .val = 0 }
#define NUXMEASURE(_s) nuxmeasure_t __measure _s = { .name = #_s , 0 }
#endif

NUXPERF(ggmlux_malloc);
NUXPERF(ggmlux_free);

NUXMEASURE(ggmlux_allocated);
NUXMEASURE(ggmlux_freed);
