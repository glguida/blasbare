/*
  The following is a C interpretation of GGML example/gpt-common.cpp

  Written by Gianluca Guida <glguida@gmail.com>

  Keeping original license.

  NB: 
   1. the sorting is excessive (not a partial sort).
   2. the tokenization is wrong. We split word using strtok. For serious uses, need to be fixed.

*/

/*
MIT License

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "util.h"
#include "cgpt-common.h"

// Implementation of strpbrk
char *strpbrk(const char *s, const char *accept) {
    while (*s != '\0') {
        const char *a = accept;
        while (*a != '\0') {
            if (*s == *a) {
                return (char *)s;  // Cast to non-const to match function signature
            }
            a++;
        }
        s++;
    }
    return NULL;
}

// Implementation of strspn
size_t strspn(const char *s, const char *accept) {
    size_t count = 0;
    while (*s != '\0') {
        if (strchr(accept, *s) == NULL) {
            break;
        }
        count++;
        s++;
    }
    return count;
}

// Basic implementation of strtok_r
char *strtok_r(char *str, const char *delim, char **saveptr) {
    if (str == NULL) {
        str = *saveptr;
    }
    
    if (str == NULL) {
        return NULL;
    }

    str += strspn(str, delim);  // Skip leading delimiters

    if (*str == '\0') {  // No token
        *saveptr = str;
        return NULL;
    }

    char *token = str;
    str = strpbrk(token, delim);  // Find next delimiter

    if (str == NULL) {  // Last token
        *saveptr = strchr(token, '\0');
    } else {
        *str = '\0';  // Terminate the token
        *saveptr = str + 1;
    }

    return token;
}

/*
  NB: FIXME. strtok is the wrong approach here, as we don't have to split words.
*/
void tokenize_words(struct vocab *v, char *text, int32_t **tokens, int *token_count)
{
    char *saveptr;
    char *word;
    *token_count = 0;
    *tokens = NULL;

    text = strdup(text);
    
    // Use our basic strtok_r to tokenize the text into words
    for (word = strtok_r(text, " \t\n", &saveptr); word != NULL; word = strtok_r(NULL, " \t\n", &saveptr)) {
        int word_len = strlen(word);
        int i = 0;
        
        while (i < word_len) {
            for (int j = word_len - 1; j >= i; j--) {
                char *cand = malloc(j - i + 2); // +1 for null terminator, +1 for safety
                if (cand == NULL) {
                    fprintf(stderr, "Memory allocation failed\n");
                    exit(1);
                }

                strncpy(cand, word + i, j - i + 1);
                cand[j - i + 1] = '\0';  // Null-terminate

                int32_t id;
		printf("Searching for candidate '%s'\n", cand);
                if (vocab_find(v, cand, &id)) {
                    // Resize tokens array
                    int32_t *tmp = realloc(*tokens, (*token_count + 1) * sizeof(int32_t));
                    if (tmp == NULL) {
                        fprintf(stderr, "Memory reallocation failed\n");
                        free(cand);
                        exit(1);
                    }
                    *tokens = tmp;
                    (*tokens)[*token_count] = id;
                    (*token_count)++;
                    i = j + 1;
                } else if (j == i) {
                    // No match for even a single character, treat as unknown token
                    fprintf(stderr, "tokenize_words: unknown token '%c'\n", word[i]);
                    i++;
                }
                free(cand);
            }
        }
    }

    free(text);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* Structure to represent a vocabulary entry */
typedef struct {
    int32_t id;
    float logit;
} sort_e;

/* Custom comparison function for qsort */
int
compare_entries (const void *a, const void *b)
{
  return ((sort_e *) b)->logit - ((sort_e *) a)->logit;
}

/* Function to perform discrete distribution sampling */
int32_t
discrete_sample (float *probs, int size, float rand_val)
{
  float sum = 0.0;
  for (int i = 0; i < size; i++)
    {
      sum += probs[i];
      if (rand_val <= sum)
        {
          return i; /* Return the index based on the random value */
        }
    }
  return size - 1; /* Fallback, shouldn't happen with proper normalization */
}

#define RAND_MAX 0x7fff

int
rand_r (unsigned long *seed)
{
  /* Constants for a simple LCG */
  const unsigned long a = 1664525;
  const unsigned long c = 1013904223;
  const unsigned long m = 0xFFFFFFFF; /* 2^32 - 1 for 32-bit integer overflow */

  /* Update the seed */
  *seed = (a * (*seed) + c) & m;

  /* Return the high 16 bits for better distribution */
  return (*seed >> 16) & 0x7FFF;
}

int32_t
gpt_sample_top_k_top_p (const float *logits,
                        int vocab_size,
                        int top_k,
                        float top_p,
                        float temp,
                        unsigned long *rng_state)
{
  /* Allocate memory for sorted logits */
  sort_e *logits_id = malloc (sizeof (sort_e) * vocab_size);
  if (!logits_id)
    exit (1); /* Handle memory allocation failure */

  /* Prepare logits for sorting */
  for (int i = 0; i < vocab_size; ++i)
    {
      logits_id[i].id = i;
      logits_id[i].logit = logits[i] / temp; /* Scale by temperature */
    }

  /* Sort logits in descending order (top K) */
  qsort (logits_id, vocab_size, sizeof (sort_e), compare_entries);

  /* Reduce to top_k if necessary */
  if (top_k < vocab_size)
    {
      vocab_size = top_k;
    }

  /* Compute probs for the top K tokens */
  float *probs = malloc (sizeof (float) * vocab_size);
  if (!probs)
    {
      free (logits_id);
      exit (1);
    }

  float maxl = -INFINITY;
  for (int i = 0; i < vocab_size; i++)
    {
      maxl = fmax (maxl, logits_id[i].logit);
    }

  float sum = 0.0;
  for (int i = 0; i < vocab_size; i++)
    {
      float p = exp (logits_id[i].logit - maxl);
      probs[i] = p;
      sum += p;
    }

  /* Normalize the probs */
  for (int i = 0; i < vocab_size; i++)
    {
      probs[i] /= sum;
    }

  /* Top-P sampling */
  if (top_p < 1.0)
    {
      float cumsum = 0.0;
      int new_size = 0;
      for (int i = 0; i < vocab_size; i++)
        {
          cumsum += probs[i];
          new_size = i + 1;
          if (cumsum >= top_p)
            break;
        }

      /* Normalize again for the reduced set */
      float normalization = 1.0 / cumsum;
      for (int i = 0; i < new_size; i++)
        {
          probs[i] *= normalization;
        }
      vocab_size = new_size;
    }

  /* Generate a random number for sampling */
  float rand_val = ((float) rand_r (rng_state)) / RAND_MAX;

  /* Sample using our discrete distribution */
  int idx = discrete_sample (probs, vocab_size, rand_val);

  int32_t result = logits_id[idx].id;

  free (logits_id);
  free (probs);

  return result;
}
