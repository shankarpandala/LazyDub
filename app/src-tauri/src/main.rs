//! Maata desktop shell (ADR-001, ADR-006).
//!
//! Starts the local Python engine with a random per-launch token, waits for its
//! `MAATA_ENGINE_READY port=… token=…` line, then points the window at the engine-served UI on
//! the loopback origin, which YouTube's embedded player accepts. The engine is killed on exit.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use tauri::{Manager, RunEvent, Url, WebviewUrl, WebviewWindowBuilder};

struct EngineProcess(Mutex<Option<Child>>);

fn random_token() -> String {
    let mut bytes = [0u8; 24];
    getrandom::fill(&mut bytes).expect("OS random source");
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// How to launch the engine:
/// - `MAATA_ENGINE_CMD` (whitespace-separated) overrides everything;
/// - debug builds run `uv run maata-engine` from the repo's `engine/`;
/// - release builds run the engine runtime installed in the app data dir (ADR-011).
fn engine_command(app: &tauri::AppHandle, token: &str) -> Result<Command, String> {
    let ui_dir: PathBuf = if cfg!(debug_assertions) {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../dist")
    } else {
        app.path().resource_dir().map_err(|e| e.to_string())?.join("ui")
    };
    let mut cmd = if let Ok(custom) = std::env::var("MAATA_ENGINE_CMD") {
        let mut parts = custom.split_whitespace();
        let mut c = Command::new(parts.next().ok_or("MAATA_ENGINE_CMD is empty")?);
        c.args(parts);
        c
    } else if cfg!(debug_assertions) {
        let engine_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../engine");
        let mut c = Command::new("uv");
        c.current_dir(engine_dir).args(["run", "--extra", "apple", "--extra", "resolve", "maata-engine"]);
        c
    } else {
        let data = app.path().app_data_dir().map_err(|e| e.to_string())?;
        let bin = data.join("engine/.venv/bin/maata-engine");
        if !bin.exists() {
            return Err("The Maata engine isn't installed yet. Run scripts/setup-mac.sh (first-run setup arrives in Phase 2).".into());
        }
        Command::new(bin)
    };
    cmd.args(["--port", "0", "--token", token, "--ui"])
        .arg(ui_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    Ok(cmd)
}

fn show_error(win: &tauri::WebviewWindow, message: &str) {
    let js = format!("window.maataSplashError && window.maataSplashError({})", serde_json::to_string(message).unwrap_or_default());
    let _ = win.eval(&js);
}

fn main() {
    let app = tauri::Builder::default()
        .manage(EngineProcess(Mutex::new(None)))
        .setup(|app| {
            let builder = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("splash.html".into()))
                .title("Maata")
                .inner_size(1440.0, 900.0)
                .min_inner_size(960.0, 640.0);
            #[cfg(target_os = "macos")]
            let builder = builder.title_bar_style(tauri::TitleBarStyle::Overlay).hidden_title(true);
            let win = builder.build()?;

            let token = random_token();
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                let mut child = match engine_command(&handle, &token).and_then(|mut c| c.spawn().map_err(|e| format!("Couldn't start the engine: {e}"))) {
                    Ok(c) => c,
                    Err(e) => return show_error(&win, &e),
                };
                let stdout = child.stdout.take();
                *handle.state::<EngineProcess>().0.lock().unwrap() = Some(child);
                let Some(stdout) = stdout else { return show_error(&win, "Engine produced no output") };
                for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                    if let Some(rest) = line.strip_prefix("MAATA_ENGINE_READY ") {
                        let port = rest.split_whitespace().find_map(|kv| kv.strip_prefix("port=")).unwrap_or("0");
                        match Url::parse(&format!("http://127.0.0.1:{port}/?token={token}")) {
                            Ok(url) => { let _ = win.navigate(url); }
                            Err(e) => show_error(&win, &e.to_string()),
                        }
                        break;
                    }
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Maata");

    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            if let Some(mut child) = handle.state::<EngineProcess>().0.lock().unwrap().take() {
                let _ = child.kill();
            }
        }
    });
}
