//! GUI: pick GGUF + `llama-server` binary, spawn backend, expose Kobold-compatible HTTP proxy.
//!
//! Build `llama-server` from `vendor/llama.cpp` (CMake), then point **Llama server exe** at
//! `build\bin\llama-server.exe` (path may vary). Settings persist under the OS config dir:
//! `hypura/kobold_gguf_gui_settings.json` (e.g. `%APPDATA%\hypura\` on Windows).

mod kobold;
mod server;
mod settings;
mod stream;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use eframe::egui;
use settings::KoboldGuiSettings;
use tokio::sync::watch;

struct GuiApp {
    settings: KoboldGuiSettings,
    log: String,
    child: Option<std::process::Child>,
    shutdown_tx: Option<watch::Sender<bool>>,
    proxy_task: Option<tokio::task::JoinHandle<anyhow::Result<()>>>,
    runtime: Option<tokio::runtime::Runtime>,
}

impl GuiApp {
    fn new() -> Self {
        Self {
            settings: KoboldGuiSettings::load(),
            log: "Stop any running servers on the chosen ports before Start.\n".to_string(),
            child: None,
            shutdown_tx: None,
            proxy_task: None,
            runtime: None,
        }
    }

    fn persist_quiet(&self) {
        if let Err(e) = self.settings.save() {
            eprintln!("kobold_gguf_gui: failed to save settings: {e:#}");
        }
    }

    fn append_log(&mut self, line: impl AsRef<str>) {
        self.log.push_str(line.as_ref());
        self.log.push('\n');
    }

    fn stop_all(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(true);
        }
        if let Some(t) = self.proxy_task.take() {
            let _ = t.abort();
        }
        self.runtime.take();
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
        self.append_log("Stopped.");
    }

    fn wait_llama_health(port: u16) -> anyhow::Result<()> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(3))
            .build()?;
        let url = format!("http://127.0.0.1:{port}/health");
        for i in 0..120 {
            match client.get(&url).send() {
                Ok(r) if r.status().is_success() => return Ok(()),
                _ => {
                    if i % 10 == 0 {
                        std::thread::sleep(Duration::from_millis(200));
                    } else {
                        std::thread::sleep(Duration::from_millis(100));
                    }
                }
            }
        }
        anyhow::bail!("llama-server did not respond on /health (port {port})");
    }

    fn start_backend(&mut self) -> anyhow::Result<()> {
        self.stop_all();

        if self.settings.gguf_path.trim().is_empty() || self.settings.llama_server_exe.trim().is_empty()
        {
            anyhow::bail!("Set GGUF path and llama-server executable.");
        }
        if !PathBuf::from(self.settings.gguf_path.trim()).exists() {
            anyhow::bail!("GGUF file not found.");
        }
        if !PathBuf::from(self.settings.llama_server_exe.trim()).exists() {
            anyhow::bail!("llama-server executable not found.");
        }

        self.settings
            .push_recent_gguf(self.settings.gguf_path.trim());
        self.persist_quiet();

        let mut cmd = Command::new(self.settings.llama_server_exe.trim());
        cmd.args([
            "-m",
            self.settings.gguf_path.trim(),
            "--port",
            &self.settings.backend_port.to_string(),
            "-ngl",
            &self.settings.ngl.to_string(),
            "-c",
            &self.settings.ctx_len.to_string(),
        ]);

        self.append_log(format!("Spawning: {:?}", cmd));
        let child = cmd.spawn().map_err(|e| anyhow::anyhow!("spawn llama-server: {e}"))?;
        self.child = Some(child);

        Self::wait_llama_health(self.settings.backend_port)?;
        self.append_log(format!(
            "llama-server OK. Kobold-compatible proxy: http://127.0.0.1:{}/api/v1/generate",
            self.settings.kobold_port
        ));

        let rt = tokio::runtime::Runtime::new()?;
        let (tx, rx) = watch::channel(false);
        self.shutdown_tx = Some(tx);

        let bind: SocketAddr = format!("127.0.0.1:{}", self.settings.kobold_port)
            .parse()
            .map_err(|e| anyhow::anyhow!("bind: {e}"))?;
        let backend = self.settings.backend_port;
        let gguf = PathBuf::from(self.settings.gguf_path.trim());
        let advertised_model = gguf
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .to_string();
        let ctx = self.settings.ctx_len;
        let meta = server::ProxyMeta {
            advertised_model,
            max_length: ctx.min(8192).max(256),
            max_context: ctx,
        };

        let task = rt.spawn(async move { server::run_proxy_server(bind, backend, rx, meta).await });

        self.proxy_task = Some(task);
        self.runtime = Some(rt);

        Ok(())
    }
}

impl eframe::App for GuiApp {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.stop_all();
        self.persist_quiet();
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Kobold-compatible GGUF proxy (llama.cpp backend)");
            ui.label("KoboldCpp互換: /api/v1/*, /api/extra/*, /api/latest/*, /v1/* → llama-server（画像・音声・TTS等は未搭載で503）。");
            ui.label(format!(
                "設定保存: {:?}",
                KoboldGuiSettings::settings_path().map(|p| p.display().to_string()).unwrap_or_else(|e| e.to_string())
            ));

            ui.horizontal(|ui| {
                ui.label("GGUF:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.settings.gguf_path).desired_width(420.0),
                );
                if ui.button("Browse…").clicked() {
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("GGUF", &["gguf"])
                        .pick_file()
                    {
                        self.settings.gguf_path = p.display().to_string();
                        self.persist_quiet();
                    }
                }
            });

            if !self.settings.recent_ggufs.is_empty() {
                ui.horizontal(|ui| {
                    ui.label("Recent GGUF:");
                    egui::ComboBox::from_id_salt("recent_gguf")
                        .selected_text("Pick from recent…")
                        .show_ui(ui, |ui| {
                            for r in self.settings.recent_ggufs.clone() {
                                if ui.selectable_label(self.settings.gguf_path == r, &r).clicked() {
                                    self.settings.gguf_path = r;
                                    self.persist_quiet();
                                }
                            }
                        });
                });
            }

            ui.horizontal(|ui| {
                ui.label("llama-server exe:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.settings.llama_server_exe)
                        .desired_width(420.0),
                );
                if ui.button("Browse…").clicked() {
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("exe", &["exe"])
                        .pick_file()
                    {
                        self.settings.llama_server_exe = p.display().to_string();
                        self.persist_quiet();
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("Backend port (llama-server):");
                ui.add(
                    egui::DragValue::new(&mut self.settings.backend_port).range(1024_u16..=65500),
                );
                ui.label("Kobold proxy port:");
                ui.add(egui::DragValue::new(&mut self.settings.kobold_port).range(1024_u16..=65500));
            });

            ui.horizontal(|ui| {
                ui.label("GPU layers (-ngl):");
                ui.add(egui::DragValue::new(&mut self.settings.ngl).range(0_u32..=999));
                ui.label("Context (-c):");
                ui.add(egui::DragValue::new(&mut self.settings.ctx_len).range(256_u32..=262144));
            });

            ui.horizontal(|ui| {
                if ui.button("Start").clicked() {
                    match self.start_backend() {
                        Ok(()) => {}
                        Err(e) => {
                            self.append_log(format!("ERROR: {e:#}"));
                            self.stop_all();
                        }
                    }
                }
                if ui.button("Stop").clicked() {
                    self.stop_all();
                }
                if ui.button("Save settings").clicked() {
                    match self.settings.save() {
                        Ok(()) => self.append_log("Settings saved."),
                        Err(e) => self.append_log(format!("Save failed: {e:#}")),
                    }
                }
            });

            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.monospace(&self.log);
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([720.0, 560.0]),
        ..Default::default()
    };
    eframe::run_native(
        "kobold_gguf_gui",
        native_options,
        Box::new(|_cc| Ok(Box::new(GuiApp::new()) as Box<dyn eframe::App>)),
    )
}
