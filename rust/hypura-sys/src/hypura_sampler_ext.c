#include "hypura_sampler_ext.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float a;
    size_t min_keep;
} hypura_sampler_top_a_ctx;

typedef struct {
    float z;
    size_t min_keep;
} hypura_sampler_tfs_ctx;

static int compare_token_logit_desc(const void * lhs, const void * rhs) {
    const llama_token_data * a = (const llama_token_data *) lhs;
    const llama_token_data * b = (const llama_token_data *) rhs;
    if (a->logit < b->logit) {
        return 1;
    }
    if (a->logit > b->logit) {
        return -1;
    }
    return 0;
}

static void ensure_sorted_logits(llama_token_data_array * cur_p) {
    if (!cur_p->sorted) {
        qsort(cur_p->data, cur_p->size, sizeof(llama_token_data), compare_token_logit_desc);
        cur_p->sorted = true;
    }
}

static void softmax_probs(llama_token_data_array * cur_p) {
    ensure_sorted_logits(cur_p);
    if (cur_p->size == 0) {
        return;
    }
    float max_l = cur_p->data[0].logit;
    float sum = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        const float p = expf(cur_p->data[i].logit - max_l);
        cur_p->data[i].p = p;
        sum += p;
    }
    if (sum <= 0.0f) {
        return;
    }
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p /= sum;
    }
}

static const char * hypura_sampler_top_a_name(const struct llama_sampler * smpl) {
    (void) smpl;
    return "hypura_top_a";
}

static void hypura_sampler_top_a_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    hypura_sampler_top_a_ctx * ctx = (hypura_sampler_top_a_ctx *) smpl->ctx;
    if (ctx == NULL || ctx->a <= 0.0f || cur_p->size <= 1) {
        return;
    }

    softmax_probs(cur_p);
    const float maxprob = cur_p->data[0].p;
    const float threshold = ctx->a * maxprob * maxprob;
    size_t last_idx = cur_p->size;
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].p < threshold && i >= ctx->min_keep) {
            last_idx = i;
            break;
        }
    }
    cur_p->size = last_idx;
}

static struct llama_sampler * hypura_sampler_top_a_clone(const struct llama_sampler * smpl) {
    const hypura_sampler_top_a_ctx * ctx = (const hypura_sampler_top_a_ctx *) smpl->ctx;
    if (ctx == NULL) {
        return NULL;
    }
    return hypura_sampler_init_top_a(ctx->a, ctx->min_keep);
}

static void hypura_sampler_top_a_free(struct llama_sampler * smpl) {
    if (smpl && smpl->ctx) {
        free(smpl->ctx);
    }
}

static struct llama_sampler_i hypura_sampler_top_a_iface = {
    /* .name              = */ hypura_sampler_top_a_name,
    /* .accept            = */ NULL,
    /* .apply             = */ hypura_sampler_top_a_apply,
    /* .reset             = */ NULL,
    /* .clone             = */ hypura_sampler_top_a_clone,
    /* .free              = */ hypura_sampler_top_a_free,
    /* .backend_init      = */ NULL,
    /* .backend_accept    = */ NULL,
    /* .backend_apply     = */ NULL,
    /* .backend_set_input = */ NULL,
};

struct llama_sampler * hypura_sampler_init_top_a(float a, size_t min_keep) {
    hypura_sampler_top_a_ctx * ctx = (hypura_sampler_top_a_ctx *) malloc(sizeof(hypura_sampler_top_a_ctx));
    if (ctx == NULL) {
        return NULL;
    }
    ctx->a = a;
    ctx->min_keep = min_keep;
    return llama_sampler_init(&hypura_sampler_top_a_iface, ctx);
}

static const char * hypura_sampler_tfs_name(const struct llama_sampler * smpl) {
    (void) smpl;
    return "hypura_tfs_z";
}

static void hypura_sampler_tfs_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    hypura_sampler_tfs_ctx * ctx = (hypura_sampler_tfs_ctx *) smpl->ctx;
    if (ctx == NULL || ctx->z >= 1.0f || cur_p->size <= 2) {
        return;
    }

    softmax_probs(cur_p);

    const size_t deriv_size = cur_p->size - 2;
    float * second_derivatives = (float *) malloc(sizeof(float) * deriv_size);
    if (second_derivatives == NULL) {
        return;
    }

    float second_sum = 0.0f;
    for (size_t i = 0; i < deriv_size; ++i) {
        const float d1 = cur_p->data[i].p - cur_p->data[i + 1].p;
        const float d2 = cur_p->data[i + 1].p - cur_p->data[i + 2].p;
        second_derivatives[i] = fabsf(d1 - d2);
        second_sum += second_derivatives[i];
    }

    if (second_sum > 1e-6f) {
        for (size_t i = 0; i < deriv_size; ++i) {
            second_derivatives[i] /= second_sum;
        }
    } else {
        const float uniform = 1.0f / (float) deriv_size;
        for (size_t i = 0; i < deriv_size; ++i) {
            second_derivatives[i] = uniform;
        }
    }

    float cum_sum = 0.0f;
    size_t last_idx = cur_p->size;
    for (size_t i = 0; i < deriv_size; ++i) {
        cum_sum += second_derivatives[i];
        if (cum_sum > ctx->z && i >= ctx->min_keep) {
            last_idx = i;
            break;
        }
    }
    cur_p->size = last_idx;
    free(second_derivatives);
}

static struct llama_sampler * hypura_sampler_tfs_clone(const struct llama_sampler * smpl) {
    const hypura_sampler_tfs_ctx * ctx = (const hypura_sampler_tfs_ctx *) smpl->ctx;
    if (ctx == NULL) {
        return NULL;
    }
    return hypura_sampler_init_tfs_z(ctx->z, ctx->min_keep);
}

static void hypura_sampler_tfs_free(struct llama_sampler * smpl) {
    if (smpl && smpl->ctx) {
        free(smpl->ctx);
    }
}

static struct llama_sampler_i hypura_sampler_tfs_iface = {
    /* .name              = */ hypura_sampler_tfs_name,
    /* .accept            = */ NULL,
    /* .apply             = */ hypura_sampler_tfs_apply,
    /* .reset             = */ NULL,
    /* .clone             = */ hypura_sampler_tfs_clone,
    /* .free              = */ hypura_sampler_tfs_free,
    /* .backend_init      = */ NULL,
    /* .backend_accept    = */ NULL,
    /* .backend_apply     = */ NULL,
    /* .backend_set_input = */ NULL,
};

struct llama_sampler * hypura_sampler_init_tfs_z(float z, size_t min_keep) {
    hypura_sampler_tfs_ctx * ctx = (hypura_sampler_tfs_ctx *) malloc(sizeof(hypura_sampler_tfs_ctx));
    if (ctx == NULL) {
        return NULL;
    }
    ctx->z = z;
    ctx->min_keep = min_keep;
    return llama_sampler_init(&hypura_sampler_tfs_iface, ctx);
}
