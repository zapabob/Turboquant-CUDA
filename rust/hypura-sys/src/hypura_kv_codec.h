#ifndef HYPURA_KV_CODEC_H
#define HYPURA_KV_CODEC_H

#include <stdint.h>
#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── KV Codec callback types (Rust → C) ──────────────────────────────────── */

/* Compress a key vector after computation.
 * Input: raw K vector (head_dim f32 values)
 * Output: compressed representation written to output buffer
 * Returns: number of bytes written to output, or -1 on error
 *
 * The codec implementation may store internal state (e.g., encoded vectors)
 * for later scoring. */
typedef int (*hypura_kv_compress_k_t)(
    void *rust_ctx,
    uint32_t layer,
    uint32_t head,
    uint32_t token_pos,
    const float *k_data,
    uint32_t head_dim,
    float *output  /* pre-allocated head_dim f32 */
);

/* Compress a value vector after computation (for paper-full-kv mode).
 * Same signature as compress_k but operates on V vectors. */
typedef int (*hypura_kv_compress_v_t)(
    void *rust_ctx,
    uint32_t layer,
    uint32_t head,
    uint32_t token_pos,
    const float *v_data,
    uint32_t head_dim,
    float *output
);

/* Compute attention scores using compressed K representations.
 * Input: query vector (head_dim f32), token range [token_start, token_end)
 * Output: scores buffer (n_tokens f32 values)
 * Returns: 0 on success, -1 on error */
typedef int (*hypura_kv_score_k_t)(
    void *rust_ctx,
    uint32_t layer,
    uint32_t head,
    const float *query,
    uint32_t head_dim,
    uint32_t token_start,
    uint32_t token_end,
    float *scores
);

/* Read V vectors for attention output computation.
 * Input: token range [token_start, token_end)
 * Output: v_buffer (n_tokens * head_dim f32 values)
 * Returns: 0 on success, -1 on error */
typedef int (*hypura_kv_read_v_t)(
    void *rust_ctx,
    uint32_t layer,
    uint32_t head,
    uint32_t head_dim,
    uint32_t token_start,
    uint32_t token_end,
    float *v_buffer
);

/* ── KV Codec configuration ──────────────────────────────────────────────── */

typedef struct {
    /* Callbacks (NULL = pass-through / no compression) */
    hypura_kv_compress_k_t compress_k;
    hypura_kv_compress_v_t compress_v;
    hypura_kv_score_k_t    score_k;
    hypura_kv_read_v_t     read_v;

    /* Opaque Rust context pointer passed to all callbacks */
    void *rust_ctx;

    /* Model dimensions */
    uint32_t num_layers;
    uint32_t num_kv_heads;
    uint32_t head_dim;

    /* Flags */
    int compress_keys;     /* 1 = apply key compression */
    int compress_values;   /* 1 = apply value compression (paper-full-kv) */
    int use_exact_score;   /* 1 = use exact scoring, 0 = use codec scoring */
} hypura_kv_codec_config;

/* ── GGML Custom Op for KV Compression ───────────────────────────────────── */

/* Create a GGML custom op node that compresses K vectors in-place.
 * This should be called during graph construction to insert a compression
 * node after the K projection and before the attention score computation.
 *
 * Parameters:
 *   ctx      - GGML context for graph building
 *   a        - Input tensor (K tensor, shape [head_dim, n_heads, n_tokens])
 *   codec    - KV codec config (must remain valid during graph execution)
 *   layer_idx - Current layer index
 *
 * Returns: Output tensor with compressed K values */
struct ggml_tensor *hypura_op_compress_k(
    struct ggml_context *ctx,
    struct ggml_tensor *a,
    const hypura_kv_codec_config *codec,
    uint32_t layer_idx
);

/* Create a GGML custom op node that compresses V vectors in-place.
 * Same semantics as compress_k but for V vectors. */
struct ggml_tensor *hypura_op_compress_v(
    struct ggml_context *ctx,
    struct ggml_tensor *a,
    const hypura_kv_codec_config *codec,
    uint32_t layer_idx
);

/* ── Runtime Codec State Management ──────────────────────────────────────── */

/* Opaque handle for runtime codec state (one per inference context). */
typedef struct hypura_kv_codec_runtime hypura_kv_codec_runtime_t;

/* Create a runtime codec state. */
hypura_kv_codec_runtime_t *hypura_kv_codec_runtime_create(
    const hypura_kv_codec_config *config
);

/* Free runtime codec state. */
void hypura_kv_codec_runtime_free(hypura_kv_codec_runtime_t *runtime);

/* Reset codec state for a new sequence (clears stored K/V vectors). */
void hypura_kv_codec_runtime_reset(hypura_kv_codec_runtime_t *runtime);

/* Get the codec config from a runtime. */
const hypura_kv_codec_config *hypura_kv_codec_runtime_get_config(
    const hypura_kv_codec_runtime_t *runtime
);

/* ── Direct Codec Operations (for Rust-side integration) ─────────────────── */

/* Compress a single K vector using the runtime codec.
 * Returns 0 on success, -1 on error. */
int hypura_kv_codec_compress_k_vec(
    hypura_kv_codec_runtime_t *runtime,
    uint32_t layer,
    uint32_t head,
    uint32_t token_pos,
    const float *k_data,
    float *output
);

/* Compress a single V vector using the runtime codec.
 * Returns 0 on success, -1 on error. */
int hypura_kv_codec_compress_v_vec(
    hypura_kv_codec_runtime_t *runtime,
    uint32_t layer,
    uint32_t head,
    uint32_t token_pos,
    const float *v_data,
    float *output
);

/* Score query against compressed K vectors.
 * Returns 0 on success, -1 on error. */
int hypura_kv_codec_score_k_vec(
    const hypura_kv_codec_runtime_t *runtime,
    uint32_t layer,
    uint32_t head,
    const float *query,
    uint32_t token_start,
    uint32_t token_end,
    float *scores
);

/* Read V vectors from runtime codec state.
 * Returns 0 on success, -1 on error. */
int hypura_kv_codec_read_v_vec(
    const hypura_kv_codec_runtime_t *runtime,
    uint32_t layer,
    uint32_t head,
    uint32_t token_start,
    uint32_t token_end,
    float *v_buffer
);

#ifdef __cplusplus
}
#endif

#endif /* HYPURA_KV_CODEC_H */
