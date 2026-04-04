#include "hypura_kv_codec.h"
#include "ggml-impl.h"
#include <stdlib.h>
#include <string.h>

/* ── Custom GGML Op: Compress K ──────────────────────────────────────────── */

/* Forward declaration for the custom op compute function */
static void hypura_compress_k_compute(
    struct ggml_tensor *dst,
    const struct ggml_tensor *src,
    int ith,
    int nth,
    void *userdata
);

static void hypura_compress_v_compute(
    struct ggml_tensor *dst,
    const struct ggml_tensor *src,
    int ith,
    int nth,
    void *userdata
);

/* Context for custom op callbacks */
typedef struct {
    const hypura_kv_codec_config *codec;
    uint32_t layer_idx;
    int is_value;  /* 0 = key, 1 = value */
} hypura_op_context;

struct ggml_tensor *hypura_op_compress_k(
    struct ggml_context *ctx,
    struct ggml_tensor *a,
    const hypura_kv_codec_config *codec,
    uint32_t layer_idx
) {
    /* Allocate persistent context for this op node */
    hypura_op_context *op_ctx = (hypura_op_context *)malloc(sizeof(hypura_op_context));
    if (!op_ctx) return NULL;
    op_ctx->codec = codec;
    op_ctx->layer_idx = layer_idx;
    op_ctx->is_value = 0;

    /* Create custom op with same shape as input */
    struct ggml_tensor *result = ggml_map_custom1(
        ctx,
        a,
        hypura_compress_k_compute,
        1,  /* n_tasks: single-threaded for now */
        op_ctx
    );

    return result;
}

struct ggml_tensor *hypura_op_compress_v(
    struct ggml_context *ctx,
    struct ggml_tensor *a,
    const hypura_kv_codec_config *codec,
    uint32_t layer_idx
) {
    hypura_op_context *op_ctx = (hypura_op_context *)malloc(sizeof(hypura_op_context));
    if (!op_ctx) return NULL;
    op_ctx->codec = codec;
    op_ctx->layer_idx = layer_idx;
    op_ctx->is_value = 1;

    struct ggml_tensor *result = ggml_map_custom1(
        ctx,
        a,
        hypura_compress_v_compute,
        1,
        op_ctx
    );

    return result;
}

/* ── Custom Op Compute Functions ─────────────────────────────────────────── */

static void hypura_compress_k_compute(
    struct ggml_tensor *dst,
    const struct ggml_tensor *src,
    int ith,
    int nth,
    void *userdata
) {
    (void)ith;
    (void)nth;

    hypura_op_context *op_ctx = (hypura_op_context *)userdata;
    if (!op_ctx || !op_ctx->codec) return;

    const hypura_kv_codec_config *codec = op_ctx->codec;
    if (!codec->compress_keys || !codec->compress_k) {
        /* No compression: pass through */
        if (dst->data != src->data) {
            memcpy(dst->data, src->data, ggml_nbytes(src));
        }
        return;
    }

    uint32_t num_heads = codec->num_kv_heads;
    uint32_t head_dim = codec->head_dim;

    /* Tensor shape: [head_dim, n_kv_heads, n_tokens] or [n_kv_heads, head_dim, n_tokens] */
    /* We iterate per-head, per-token */
    const float *src_data = (const float *)src->data;
    float *dst_data = (float *)dst->data;

    /* Determine layout from tensor dimensions */
    int64_t ne0 = src->ne[0];  /* head_dim or n_kv_heads */
    int64_t ne1 = src->ne[1];  /* n_kv_heads or head_dim */
    int64_t ne2 = src->ne[2];  /* n_tokens */

    for (int64_t token = 0; token < ne2; token++) {
        for (int64_t head = 0; head < (int64_t)num_heads; head++) {
            const float *k_in;
            float *k_out;

            /* Standard layout: [head_dim, n_kv_heads, n_tokens] */
            if (ne0 == (int64_t)head_dim) {
                k_in = &src_data[token * num_heads * head_dim + head * head_dim];
                k_out = &dst_data[token * num_heads * head_dim + head * head_dim];
            } else {
                /* Alternative layout: [n_kv_heads, head_dim, n_tokens] */
                k_in = &src_data[token * num_heads * head_dim * 1 + head * head_dim];
                k_out = &dst_data[token * num_heads * head_dim * 1 + head * head_dim];
            }

            codec->compress_k(
                codec->rust_ctx,
                op_ctx->layer_idx,
                (uint32_t)head,
                (uint32_t)token,
                k_in,
                head_dim,
                k_out
            );
        }
    }
}

static void hypura_compress_v_compute(
    struct ggml_tensor *dst,
    const struct ggml_tensor *src,
    int ith,
    int nth,
    void *userdata
) {
    (void)ith;
    (void)nth;

    hypura_op_context *op_ctx = (hypura_op_context *)userdata;
    if (!op_ctx || !op_ctx->codec) return;

    const hypura_kv_codec_config *codec = op_ctx->codec;
    if (!codec->compress_values || !codec->compress_v) {
        if (dst->data != src->data) {
            memcpy(dst->data, src->data, ggml_nbytes(src));
        }
        return;
    }

    uint32_t num_heads = codec->num_kv_heads;
    uint32_t head_dim = codec->head_dim;

    const float *src_data = (const float *)src->data;
    float *dst_data = (float *)dst->data;

    int64_t ne2 = src->ne[2];  /* n_tokens */

    for (int64_t token = 0; token < ne2; token++) {
        for (int64_t head = 0; head < (int64_t)num_heads; head++) {
            const float *v_in = &src_data[token * num_heads * head_dim + head * head_dim];
            float *v_out = &dst_data[token * num_heads * head_dim + head * head_dim];

            codec->compress_v(
                codec->rust_ctx,
                op_ctx->layer_idx,
                (uint32_t)head,
                (uint32_t)token,
                v_in,
                head_dim,
                v_out
            );
        }
    }
}

/* ── Runtime Codec State ─────────────────────────────────────────────────── */

struct hypura_kv_codec_runtime {
    hypura_kv_codec_config config;
    /* Future: add per-layer, per-head encoded vector storage here */
};

hypura_kv_codec_runtime_t *hypura_kv_codec_runtime_create(
    const hypura_kv_codec_config *config
) {
    if (!config) return NULL;

    hypura_kv_codec_runtime_t *rt = (hypura_kv_codec_runtime_t *)calloc(1, sizeof(hypura_kv_codec_runtime_t));
    if (!rt) return NULL;

    rt->config = *config;

    return rt;
}

void hypura_kv_codec_runtime_free(hypura_kv_codec_runtime_t *runtime) {
    if (runtime) {
        free(runtime);
    }
}

void hypura_kv_codec_runtime_reset(hypura_kv_codec_runtime_t *runtime) {
    /* Future: clear stored encoded vectors */
    (void)runtime;
}

const hypura_kv_codec_config *hypura_kv_codec_runtime_get_config(
    const hypura_kv_codec_runtime_t *runtime
) {
    return runtime ? &runtime->config : NULL;
}

/* ── Direct Codec Operations ─────────────────────────────────────────────── */

int hypura_kv_codec_compress_k_vec(
    hypura_kv_codec_runtime_t *runtime,
    uint32_t layer,
    uint32_t head,
    uint32_t token_pos,
    const float *k_data,
    float *output
) {
    if (!runtime || !runtime->config.compress_k) return -1;

    return runtime->config.compress_k(
        runtime->config.rust_ctx,
        layer,
        head,
        token_pos,
        k_data,
        runtime->config.head_dim,
        output
    );
}

int hypura_kv_codec_compress_v_vec(
    hypura_kv_codec_runtime_t *runtime,
    uint32_t layer,
    uint32_t head,
    uint32_t token_pos,
    const float *v_data,
    float *output
) {
    if (!runtime || !runtime->config.compress_v) return -1;

    return runtime->config.compress_v(
        runtime->config.rust_ctx,
        layer,
        head,
        token_pos,
        v_data,
        runtime->config.head_dim,
        output
    );
}

int hypura_kv_codec_score_k_vec(
    const hypura_kv_codec_runtime_t *runtime,
    uint32_t layer,
    uint32_t head,
    const float *query,
    uint32_t token_start,
    uint32_t token_end,
    float *scores
) {
    if (!runtime || !runtime->config.score_k) return -1;

    return runtime->config.score_k(
        runtime->config.rust_ctx,
        layer,
        head,
        query,
        runtime->config.head_dim,
        token_start,
        token_end,
        scores
    );
}

int hypura_kv_codec_read_v_vec(
    const hypura_kv_codec_runtime_t *runtime,
    uint32_t layer,
    uint32_t head,
    uint32_t token_start,
    uint32_t token_end,
    float *v_buffer
) {
    if (!runtime || !runtime->config.read_v) return -1;

    return runtime->config.read_v(
        runtime->config.rust_ctx,
        layer,
        head,
        runtime->config.head_dim,
        token_start,
        token_end,
        v_buffer
    );
}
