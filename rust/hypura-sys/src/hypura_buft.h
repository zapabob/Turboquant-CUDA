#ifndef HYPURA_BUFT_H
#define HYPURA_BUFT_H

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Callbacks from C into Rust during model loading */
typedef void (*hypura_on_tensor_loaded_t)(void *rust_ctx, const char *name, size_t offset, size_t size, void *buffer_base);
typedef void (*hypura_on_tensor_init_t)(void *rust_ctx, struct ggml_tensor *tensor, const char *name);

/* Create a custom buffer type for NVMe-tier tensors.
 * Tensors assigned to this buffer type are allocated in page-aligned RAM.
 * Callbacks notify Rust of tensor metadata during model loading. */
ggml_backend_buffer_type_t hypura_buft_create(
    hypura_on_tensor_loaded_t on_tensor_loaded,
    hypura_on_tensor_init_t on_tensor_init,
    void *rust_ctx
);

/* Free the buffer type (does not free buffers — ggml handles that). */
void hypura_buft_free(ggml_backend_buffer_type_t buft);

/* Get the raw base pointer of a buffer allocated by this buffer type. */
void *hypura_buffer_get_base_ptr(ggml_backend_buffer_t buffer);

/* --- Pool buffer API for expert-streaming mode --- */

/* Initialize a pool buffer for expert-streaming. Allocates a separate mmap region
 * of pool_size bytes. Does NOT free the original loading buffer. Returns 0 on success. */
int hypura_buffer_init_pool(ggml_backend_buffer_t buffer, size_t pool_size);

/* Free the original loading buffer (call after rewriting tensor->data to pool slots).
 * The pool buffer remains active. */
void hypura_buffer_release_loading_buffer(ggml_backend_buffer_t buffer);

/* Get pool base pointer (NULL if pool not initialized). */
void *hypura_buffer_get_pool_base(ggml_backend_buffer_t buffer);

/* Get pool size in bytes (0 if pool not initialized). */
size_t hypura_buffer_get_pool_size(ggml_backend_buffer_t buffer);

/* Get the most recently allocated buffer handle from this buffer type. */
ggml_backend_buffer_t hypura_buft_get_last_buffer(ggml_backend_buffer_type_t buft);

#ifdef __cplusplus
}
#endif

#endif /* HYPURA_BUFT_H */
