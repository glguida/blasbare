# GGML in NUX kernel support library.

This is a library required to support compiling GGML in kernel mode
in a NUX kernel.

Contains a bare minimum stdlibc++ (from uClibc++ 2.5) to support
vector and string.

Also contains extensions to the libec for more complex functions that
are required by GGML but not useful in a generic kernel environment
(qsort, strcmp).

Furthermore, defines the memory allocations and thread model functions
to be used by GGML.
