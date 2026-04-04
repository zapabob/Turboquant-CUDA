use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Memory/storage tier for tensor placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StorageTier {
    /// Metal GPU (fastest, most limited)
    Gpu,
    /// System RAM
    Ram,
    /// NVMe SSD (largest, slowest)
    Nvme,
}

/// User experience classification based on predicted tok/s.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExperienceTier {
    /// >10 tok/s — real-time chat
    Fast,
    /// 2-10 tok/s — usable for coding, Q&A
    Usable,
    /// <2 tok/s — background tasks only
    Slow,
}

impl ExperienceTier {
    pub fn from_tok_per_sec(tps: f64) -> Self {
        if tps > 10.0 {
            Self::Fast
        } else if tps > 2.0 {
            Self::Usable
        } else {
            Self::Slow
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Fast => "FAST",
            Self::Usable => "USABLE",
            Self::Slow => "SLOW",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Fast => "suitable for real-time chat",
            Self::Usable => "suitable for coding, Q&A. Not real-time chat.",
            Self::Slow => "background tasks only",
        }
    }
}

/// Inference mode determined by model size, MoE structure, and hardware capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceMode {
    /// All tensors fit in GPU+RAM. No NVMe I/O.
    FullResident,
    /// NVMe spill is small enough to keep loaded after first pass.
    KeepResident,
    /// MoE model: non-expert tensors resident, experts streamed from NVMe on demand.
    ExpertStreaming,
    /// Dense model: attention + norms resident, FFN tensors streamed from NVMe on demand.
    DenseFfnStreaming,
    /// Ultra-sparse MoE: active working set fits in RAM. Use mmap, let OS page cache
    /// handle sparsity. No pool, no eval callback — faster than expert-streaming when
    /// activation ratio is very low (e.g. 512 experts, 10 active = 2%).
    SparseMoeMmap,
    /// Heavy NVMe spill: all NVMe layers streamed, loaded/released per token.
    FullStreaming,
}

/// Complete placement plan for a model on specific hardware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementPlan {
    pub model_id: String,
    pub hardware_profile_hash: String,
    /// tensor_name → tier
    pub tier_assignments: HashMap<String, StorageTier>,
    pub prefetch_schedule: PrefetchSchedule,
    pub estimated_tok_per_sec: f64,
    pub estimated_time_to_first_token: f64,
    pub kv_cache_plan: KvCachePlan,
    pub experience_tier: ExperienceTier,
    pub inference_mode: InferenceMode,
}

/// Schedule for prefetching tensors from slower tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchSchedule {
    /// For each layer, which tensors to begin loading during the previous layer's compute.
    pub layer_prefetches: Vec<Vec<PrefetchOp>>,
}

/// A single prefetch operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchOp {
    pub tensor_name: String,
    pub source_tier: StorageTier,
    pub target_tier: StorageTier,
    pub size_bytes: u64,
    /// How many layers ahead this prefetch should start.
    pub lead_time_layers: u32,
}

/// KV cache quantization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvQuantization {
    F16,
    Q8_0,
    Q4_0,
}

impl KvQuantization {
    /// Memory scale factor relative to F16.
    pub fn memory_scale(&self) -> f64 {
        match self {
            Self::F16 => 1.0,
            Self::Q8_0 => 0.53,
            Self::Q4_0 => 0.28,
        }
    }
}

/// Plan for KV cache allocation across tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCachePlan {
    pub hot_window_tokens: u32,
    pub warm_window_tokens: u32,
    pub hot_tier: StorageTier,
    pub warm_tier: StorageTier,
    pub hot_bytes: u64,
    pub warm_bytes: u64,
    /// Auto-selected KV quantization (None = F16 default).
    pub kv_quantization: Option<KvQuantization>,
}

/// Summary of placement for display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementSummary {
    pub layers_on_gpu: u32,
    pub layers_in_ram: u32,
    pub layers_on_nvme: u32,
    pub total_gpu_bytes: u64,
    pub total_ram_bytes: u64,
    pub total_nvme_bytes: u64,
}

/// Unified memory budget estimation for Apple Silicon.
///
/// Replaces the inconsistent per-site overhead estimates (1 GB / 2 GB / 2.5 GB)
/// with a single computation based on actual model parameters.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    pub total_ram: u64,
    /// GPU tensor bytes committed in physical memory.
    /// With use_mmap=true: ~60% (demand paging). With use_mmap=false: 100%.
    pub gpu_committed: u64,
    /// KV cache size in bytes (F16 baseline, before quantization).
    pub kv_cache_bytes: u64,
    /// Metal compute buffer overhead (~300 MB measured + framework).
    pub metal_overhead: u64,
    /// OS + page cache reservation.
    pub os_overhead: u64,
    /// Pool buffer size (if any).
    pub pool_bytes: u64,
    /// Resident FFN buffer size (if any).
    pub resident_bytes: u64,
    /// Total committed = gpu + kv + metal + os + pool + resident.
    pub total_committed: u64,
    /// Available for new allocations (total_ram - total_committed).
    pub available: u64,
}

/// Measured Metal compute buffer overhead. The benchmark showed ~273 MB MTL0 +
/// ~33 MB CPU compute = ~306 MB. Round up to 512 MB for safety.
const METAL_COMPUTE_OVERHEAD: u64 = 512 * (1 << 20);

/// OS + page cache reservation. macOS kernel + WindowServer + system daemons
/// typically consume 3-4 GB on Apple Silicon.
const OS_RESERVED: u64 = 3 * (1 << 30);

impl MemoryBudget {
    /// Compute memory budget from model and hardware parameters.
    ///
    /// `gpu_tensor_bytes`: total bytes of tensors on Metal GPU.
    /// `use_mmap`: true if Metal uses mmap (60% commit), false if fread (100%).
    /// `num_layers`, `num_kv_heads`, `head_dim`, `context_length`: for KV cache.
    /// `kv_quant`: optional KV quantization (reduces KV cache size).
    pub fn compute(
        total_ram: u64,
        gpu_tensor_bytes: u64,
        use_mmap: bool,
        num_layers: u32,
        num_kv_heads: u32,
        head_dim: u64,
        context_length: u32,
        kv_quant: Option<KvQuantization>,
    ) -> Self {
        let gpu_committed = if use_mmap {
            gpu_tensor_bytes * 60 / 100
        } else {
            gpu_tensor_bytes
        };

        // KV cache: 2 (K+V) × layers × kv_heads × head_dim × 2 (FP16) × context
        let kv_raw =
            2 * num_layers as u64 * num_kv_heads as u64 * head_dim * 2 * context_length as u64;
        let kv_cache_bytes = match kv_quant {
            Some(q) => (kv_raw as f64 * q.memory_scale()) as u64,
            None => kv_raw,
        };

        let total_committed = gpu_committed + kv_cache_bytes + METAL_COMPUTE_OVERHEAD + OS_RESERVED;

        let available = total_ram.saturating_sub(total_committed);

        Self {
            total_ram,
            gpu_committed,
            kv_cache_bytes,
            metal_overhead: METAL_COMPUTE_OVERHEAD,
            os_overhead: OS_RESERVED,
            pool_bytes: 0,
            resident_bytes: 0,
            total_committed,
            available,
        }
    }

    /// Update with pool buffer allocation.
    pub fn with_pool(mut self, pool_bytes: u64) -> Self {
        self.pool_bytes = pool_bytes;
        self.total_committed += pool_bytes;
        self.available = self.total_ram.saturating_sub(self.total_committed);
        self
    }

    /// Update with resident buffer allocation.
    pub fn with_resident(mut self, resident_bytes: u64) -> Self {
        self.resident_bytes = resident_bytes;
        self.total_committed += resident_bytes;
        self.available = self.total_ram.saturating_sub(self.total_committed);
        self
    }

    /// Compute dynamic pool slot count based on available memory.
    /// Returns a slot count clamped between `min_slots` and `max_slots`.
    pub fn pool_slots(&self, slot_size: u64, min_slots: usize, max_slots: usize) -> usize {
        if slot_size == 0 {
            return min_slots;
        }
        let slots_from_memory = (self.available / slot_size) as usize;
        slots_from_memory.clamp(min_slots, max_slots)
    }
}
