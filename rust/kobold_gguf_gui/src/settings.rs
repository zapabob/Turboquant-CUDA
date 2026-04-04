//! Persisted GUI state: GGUF path, llama-server exe, ports, recent GGUF list.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Max entries kept for the "Recent GGUF" quick picker.
pub const MAX_RECENT_GGUF: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct KoboldGuiSettings {
    pub gguf_path: String,
    pub llama_server_exe: String,
    pub backend_port: u16,
    pub kobold_port: u16,
    pub ngl: u32,
    pub ctx_len: u32,
    /// Most recent first.
    pub recent_ggufs: Vec<String>,
}

impl Default for KoboldGuiSettings {
    fn default() -> Self {
        Self {
            gguf_path: String::new(),
            llama_server_exe: String::new(),
            backend_port: 8081,
            kobold_port: 5001,
            ngl: 99,
            ctx_len: 4096,
            recent_ggufs: Vec::new(),
        }
    }
}

impl KoboldGuiSettings {
    pub fn settings_path() -> anyhow::Result<PathBuf> {
        let base = dirs::config_dir()
            .ok_or_else(|| anyhow::anyhow!("config directory not available"))?
            .join("hypura");
        fs::create_dir_all(&base)?;
        Ok(base.join("kobold_gguf_gui_settings.json"))
    }

    pub fn load() -> Self {
        Self::settings_path()
            .ok()
            .and_then(|p| fs::read_to_string(&p).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let p = Self::settings_path()?;
        let tmp = p.with_extension("tmp");
        fs::write(&tmp, serde_json::to_string_pretty(self)?)?;
        fs::rename(&tmp, &p)?;
        Ok(())
    }

    pub fn push_recent_gguf(&mut self, path: &str) {
        let p = path.trim().to_string();
        if p.is_empty() {
            return;
        }
        self.recent_ggufs.retain(|x| x != &p);
        self.recent_ggufs.insert(0, p);
        self.recent_ggufs.truncate(MAX_RECENT_GGUF);
    }
}
