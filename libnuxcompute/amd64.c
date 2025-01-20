#include <stdio.h>

void nc_cpu_init (void)
{
  /*
    Start AVX and SSE. (INTEL SPECIFIC)
  */

  printf("Enabling SSE/AVX...");
  asm volatile (
		"mov %%cr0, %%rax\n"
		"and $~0x8, %%rax\n" /* Clear TS in case. */
		"and $~0x4, %%rax\n" /* Clear EM in case. */
		"mov %%rax, %%cr0\n"

		"mov %%cr0, %%rax\n"
		"or $0x2, %%rax\n"  /* Set MP */
		"mov %%rax, %%cr0\n"

		"mov %%cr4, %%rax\n"
		"or $0x200, %%rax\n"   /* Set OSFXSR */
		"or $0x400, %%rax\n"   /* Set OSXMMEXCPT */
		"or $0x40000, %%rax\n" /* Set OSXSAVE */
		"mov %%rax, %%cr4\n"

		"xor %%ecx, %%ecx\n" /* XCR0 */
		"mov $0x7, %%eax\n"  /* Enable x87, SSE, AVX */
		"xor %%edx, %%edx\n" /* Upper bits empty. */
		"xsetbv\n"
		::: "rax", "rcx", "rdx");
  printf("done\n");
}
