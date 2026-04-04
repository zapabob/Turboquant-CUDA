// Custom GGML backend for tiered tensor placement.
//
// Phase 3 decision: using standard llama.cpp API with n_gpu_layers derived
// from the PlacementPlan. Custom backend deferred to Phase 3b when we need
// per-tensor NVMe/RAM placement beyond llama.cpp's layer-granularity offload.
