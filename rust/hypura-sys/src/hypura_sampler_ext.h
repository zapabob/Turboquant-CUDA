#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

struct llama_sampler * hypura_sampler_init_top_a(float a, size_t min_keep);
struct llama_sampler * hypura_sampler_init_tfs_z(float z, size_t min_keep);

#ifdef __cplusplus
}
#endif
