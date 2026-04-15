use std::env;
use std::path::PathBuf;

/// Resolve a zapabob/llama.cpp-compatible checkout: env override, then
/// workspace-local `vendor/`, then hub layout `rust/../vendor/`.
fn resolve_llama_cpp_dir(manifest_dir: &str) -> PathBuf {
    for key in ["LLAMA_CPP_DIR", "HYPURA_LLAMA_CPP_DIR"] {
        if let Ok(p) = env::var(key) {
            let path = PathBuf::from(p.trim());
            if path.is_dir() {
                return dunce::canonicalize(&path).unwrap_or(path);
            }
        }
    }
    let candidates = [
        PathBuf::from(manifest_dir).join("../vendor/llama.cpp"),
        PathBuf::from(manifest_dir).join("../../vendor/llama.cpp"),
    ];
    for c in &candidates {
        if c.is_dir() {
            return dunce::canonicalize(c).expect("canonicalize llama.cpp path");
        }
    }
    panic!(
        "zapabob/llama.cpp-compatible checkout not found. Set LLAMA_CPP_DIR or HYPURA_LLAMA_CPP_DIR, or place vendor/llama.cpp (tried {:?})",
        candidates
    );
}

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let llama_dir = resolve_llama_cpp_dir(&manifest_dir);
    // dunce::canonicalize strips the \\?\ UNC prefix that std::fs::canonicalize
    // adds on Windows, which would otherwise cause MSBuild to reject source paths.
    let llama_dir = dunce::canonicalize(&llama_dir).unwrap_or(llama_dir);

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let use_metal = target_os == "macos";
    let use_cuda = !use_metal && cuda_is_available();
    configure_cmake_num_jobs(&target_os);

    // ── Build llama.cpp via cmake ────────────────────────────────────────────
    let mut cmake_config = cmake::Config::new(&llama_dir);
    // cmake-rs may default to a preview VS generator (e.g. VS 18 2026) that is not
    // installed. Prefer VS 2022 when the user has not set CMAKE_GENERATOR.
    if target_os == "windows" && env::var("CMAKE_GENERATOR").is_err() {
        cmake_config.generator("Visual Studio 17 2022");
    }
    cmake_config
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL")
        // Windows try_run cleanup is more reproducible without opportunistic sccache handles.
        .define("GGML_CCACHE", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("GGML_CPU", "ON")
        .define("GGML_BLAS", "OFF");

    if use_metal {
        // macOS / Apple Silicon — Metal GPU
        cmake_config
            .define("GGML_METAL", "ON")
            .define("GGML_METAL_EMBED_LIBRARY", "ON")
            .define("GGML_CUDA", "OFF")
            .define("GGML_OPENMP", "OFF");
    } else if use_cuda {
        // Windows / WSL2 / Linux — NVIDIA CUDA
        // Target RTX 20xx (sm_75) and up through RTX 50xx / H100 (sm_120).
        // "native" detects only the current machine's GPU; a fixed list enables
        // building a binary that runs on multiple NVIDIA generations.
        let cuda_arches =
            env::var("HYPURA_CUDA_ARCHITECTURES").unwrap_or_else(|_| "75;86;89;90".to_string());

        cmake_config
            .define("GGML_METAL", "OFF")
            .define("GGML_CUDA", "ON")
            .define("GGML_OPENMP", "ON")
            .define("CMAKE_CUDA_ARCHITECTURES", cuda_arches);

        if let Some(nvcc) = find_nvcc() {
            cmake_config.define("CMAKE_CUDA_COMPILER", nvcc.display().to_string());
        }
        if let Some(cuda_root) = get_cuda_root() {
            cmake_config.define("CUDAToolkit_ROOT", cuda_root.display().to_string());
        }
    } else {
        // CPU-only fallback
        cmake_config
            .define("GGML_METAL", "OFF")
            .define("GGML_CUDA", "OFF")
            .define("GGML_OPENMP", "ON");
    }

    let dst = cmake_config.build();
    let lib_dir = dst.join("lib");

    // ── Link the static libraries ────────────────────────────────────────────
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");

    if use_metal {
        println!("cargo:rustc-link-lib=static=ggml-metal");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=c++");
    } else if use_cuda {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
        if let Some(lib_path) = get_cuda_lib_path() {
            println!("cargo:rustc-link-search=native={}", lib_path.display());
        }
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudart");
        if target_os == "linux" {
            println!("cargo:rustc-link-lib=stdc++");
        }
        // Windows: MSVC links its C++ runtime automatically.
    } else if target_os == "linux" {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // Propagate feature flags to Rust code via cfg()
    if use_metal {
        println!("cargo:rustc-cfg=hypura_metal");
    } else if use_cuda {
        println!("cargo:rustc-cfg=hypura_cuda");
    }

    // ── Compile the custom GGML buffer type C shim ───────────────────────────
    let src_dir = PathBuf::from(&manifest_dir).join("src");
    let include_ggml_internal = llama_dir.join("ggml/src");

    let mut cc_build = cc::Build::new();
    cc_build
        .file(src_dir.join("hypura_buft.c"))
        .file(src_dir.join("hypura_kv_codec.c"))
        .file(src_dir.join("hypura_sampler_ext.c"))
        .include(llama_dir.join("include"))
        .include(llama_dir.join("ggml/include"))
        .include(&include_ggml_internal)
        .include(&src_dir);

    // MSVC doesn't accept -std=c11; GCC/Clang do.
    if target_os != "windows" {
        cc_build.flag("-std=c11");
    }
    cc_build.compile("hypura_buft");

    println!("cargo:rerun-if-changed=src/hypura_buft.c");
    println!("cargo:rerun-if-changed=src/hypura_buft.h");
    println!("cargo:rerun-if-changed=src/hypura_kv_codec.c");
    println!("cargo:rerun-if-changed=src/hypura_kv_codec.h");
    println!("cargo:rerun-if-changed=src/hypura_sampler_ext.c");
    println!("cargo:rerun-if-changed=src/hypura_sampler_ext.h");

    // ── Generate Rust bindings via bindgen ───────────────────────────────────
    let include_llama = llama_dir.join("include");
    let include_ggml = llama_dir.join("ggml/include");

    let bindings = bindgen::Builder::default()
        .header(
            PathBuf::from(&manifest_dir)
                .join("wrapper.h")
                .to_str()
                .unwrap()
                .to_string(),
        )
        .clang_arg(format!("-I{}", include_llama.display()))
        .clang_arg(format!("-I{}", include_ggml.display()))
        .clang_arg(format!("-I{}", src_dir.display()))
        .allowlist_function("llama_.*")
        .allowlist_function("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_function("hypura_.*")
        .allowlist_type("llama_.*")
        .allowlist_type("ggml_.*")
        .allowlist_type("gguf_.*")
        .allowlist_type("hypura_.*")
        .allowlist_var("LLAMA_.*")
        .allowlist_var("GGML_.*")
        .allowlist_var("GGUF_.*")
        .derive_debug(true)
        .derive_default(true)
        .layout_tests(false)
        .generate()
        .expect("Failed to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Failed to write bindings");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!(
        "cargo:rerun-if-changed={}",
        llama_dir.join("include").display()
    );
    println!("cargo:rerun-if-changed={}", llama_dir.join("src").display());
    println!(
        "cargo:rerun-if-changed={}",
        llama_dir.join("ggml").display()
    );
}

// ── CUDA detection helpers ────────────────────────────────────────────────────

fn configure_cmake_num_jobs(target_os: &str) {
    if target_os != "windows" {
        return;
    }

    if let Ok(value) = env::var("HYPURA_CMAKE_NUM_JOBS") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            env::set_var("NUM_JOBS", trimmed);
            println!(
                "cargo:warning=Using HYPURA_CMAKE_NUM_JOBS={trimmed} for llama.cpp CMake builds."
            );
            return;
        }
    }

    if env::var_os("NUM_JOBS").is_none() {
        env::set_var("NUM_JOBS", "1");
        println!(
            "cargo:warning=Defaulting NUM_JOBS=1 for Windows llama.cpp CMake builds to avoid MSVC link instability. Set HYPURA_CMAKE_NUM_JOBS to override."
        );
    }
}

fn cuda_is_available() -> bool {
    // Explicit opt-out
    if env::var("HYPURA_NO_CUDA").is_ok() {
        return false;
    }
    // Explicit opt-in (useful in CI or when auto-detection fails)
    if env::var("HYPURA_CUDA").is_ok() {
        return true;
    }
    get_cuda_root().is_some()
}

/// Return the CUDA toolkit root, trying common locations.
fn get_cuda_root() -> Option<PathBuf> {
    // Set by the Windows CUDA installer or by the user
    if let Ok(p) = env::var("CUDA_PATH") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }

    // Linux / WSL2 default
    for candidate in &["/usr/local/cuda", "/usr/cuda"] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return Some(p);
        }
    }

    // If nvcc is on PATH, try to derive the root from it
    if let Some(nvcc) = find_nvcc() {
        if let Some(bin) = nvcc.parent() {
            if let Some(root) = bin.parent() {
                return Some(root.to_path_buf());
            }
        }
    }

    None
}

fn get_cuda_lib_path() -> Option<PathBuf> {
    let root = get_cuda_root()?;
    for sub in &["lib64", "lib/x64", "lib"] {
        let p = root.join(sub);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn find_nvcc() -> Option<PathBuf> {
    // Check well-known paths first to avoid PATH-injection
    let candidates = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/cuda/bin/nvcc",
        // Windows: CUDA_PATH is checked above; if we reach here, fall back to PATH
    ];
    for c in &candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
    }

    // Last-resort: check that `nvcc` is runnable
    let ok = std::process::Command::new("nvcc")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if ok {
        return Some(PathBuf::from("nvcc")); // rely on PATH
    }

    None
}
