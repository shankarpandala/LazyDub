//! Maata desktop shell (ADR-001, ADR-006; OFFLINE-RENDER §5).
//!
//! Starts the local Python engine with a random per-launch token, waits for its
//! `MAATA_ENGINE_READY port=… token=…` line, then points the window at the engine-served UI on
//! the loopback origin. The engine runs the dubbing jobs, so on macOS closing the window only hides
//! it (the jobs go on; the Dock icon shows it again). Quitting closes the engine's stdin, its
//! lifeline, so it stops the running job gracefully (it continues at the next launch), and kills it
//! only if it hasn't exited within QUIT_WAIT. An engine that dies on its own (an out-of-memory kill,
//! an MPS crash) is started again, with a new token and port, at most RESTARTS times in
//! RESTART_WINDOW; its startup recovery continues the job.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::VecDeque;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{Manager, RunEvent, Url, WebviewUrl, WebviewWindow, WebviewWindowBuilder};

/// How long a quit waits for the engine to stop gracefully before killing it. The engine itself
/// exits within 4 s of losing its stdin (server.py QUIT_WAIT), so the kill is a backstop.
const QUIT_WAIT: Duration = Duration::from_secs(5);
const QUIT_POLL: Duration = Duration::from_millis(100);
/// At most this many restarts of an engine that died on its own within RESTART_WINDOW.
const RESTARTS: usize = 3;
const RESTART_WINDOW: Duration = Duration::from_secs(600);
const SETUP_RUNTIME: &str = "Quit Maata first, run Setup Maata.command from the matching installer, then reopen Maata.";

/// The engine process, and whether Maata is quitting (then an engine that stops isn't restarted).
struct Engine {
    child: Mutex<Option<Child>>,
    quitting: AtomicBool,
}

fn random_token() -> String {
    let mut bytes = [0u8; 24];
    getrandom::fill(&mut bytes).expect("OS random source");
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A release id is a full SHA-256, never a path or an arbitrary version label.
fn runtime_id(bytes: &[u8], require_schema: bool) -> Result<String, ()> {
    let doc: serde_json::Value = serde_json::from_slice(bytes).map_err(|_| ())?;
    if require_schema && doc.get("schema").and_then(serde_json::Value::as_u64) != Some(1) {
        return Err(());
    }
    let id = doc.get("runtime_id").and_then(serde_json::Value::as_str).ok_or(())?;
    if id.len() != 64 || !id.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(());
    }
    Ok(id.to_owned())
}

/// Never silently launch an engine left by a different installer or an old checkout shim.
fn installed_engine(resources: &Path, data: &Path) -> Result<PathBuf, String> {
    let bundled = fs::read(resources.join("runtime/release.json"))
        .ok().and_then(|bytes| runtime_id(&bytes, true).ok())
        .ok_or_else(|| format!("Maata's bundled runtime information is missing or invalid. {SETUP_RUNTIME}"))?;
    let installed = fs::read(data.join("engine/installed-runtime.json"))
        .ok().and_then(|bytes| runtime_id(&bytes, false).ok())
        .ok_or_else(|| format!("The matching Maata engine is not installed. {SETUP_RUNTIME}"))?;
    if installed != bundled {
        return Err(format!("The installed engine belongs to a different Maata release. {SETUP_RUNTIME}"));
    }
    let bin = data.join("engine/.venv/bin/maata-engine");
    if !bin.is_file() {
        return Err(format!("The installed Maata engine launcher is missing. {SETUP_RUNTIME}"));
    }
    Ok(bin)
}

/// Finder/Dock launches omit shell startup files. Preserve custom lookup locations after predictable native bins.
fn native_path(data: &Path, home: &Path, inherited: Option<&OsStr>) -> Result<OsString, String> {
    let mut paths = vec![data.join("engine/.venv/bin"), data.join("bin"), home.join(".local/bin"),
        home.join(".deno/bin"), PathBuf::from("/opt/homebrew/bin"), PathBuf::from("/usr/local/bin"),
        PathBuf::from("/usr/bin"), PathBuf::from("/bin"), PathBuf::from("/usr/sbin"), PathBuf::from("/sbin")];
    if let Some(inherited) = inherited {
        for path in std::env::split_paths(inherited) {
            if !path.as_os_str().is_empty() && !paths.contains(&path) {
                paths.push(path);
            }
        }
    }
    std::env::join_paths(paths).map_err(|e| format!("Couldn't configure the engine's tool paths: {e}"))
}

/// How to launch the engine:
/// - `MAATA_ENGINE_CMD` (whitespace-separated) overrides everything;
/// - debug builds run `uv run maata-engine` from the repo's `engine/`;
/// - release builds require the installed engine stamp to match their bundled runtime (ADR-011).
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
        let resources = app.path().resource_dir().map_err(|e| e.to_string())?;
        let home = app.path().home_dir().map_err(|e| e.to_string())?;
        let mut command = Command::new(installed_engine(&resources, &data)?);
        command.current_dir(data.join("engine"));
        command.env("PATH", native_path(&data, &home, std::env::var_os("PATH").as_deref())?);
        command
    };
    // The engine exits when its stdin closes. The write end lives in the stored `Child`, so the
    // engine goes away with the shell even if the shell is killed without running its exit hook.
    cmd.args(["--port", "0", "--token", token, "--stdin-lifeline", "--ui"])
        .arg(ui_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    Ok(cmd)
}

/// The splash page (still loaded) says why the engine couldn't start.
fn show_error(win: &WebviewWindow, message: &str) {
    let js = format!("window.maataSplashError && window.maataSplashError({})", serde_json::to_string(message).unwrap_or_default());
    let _ = win.eval(&js);
}

/// The window, hidden or not, goes back to the splash page, which shows `message` (its `?error=`).
fn show_error_page(win: &WebviewWindow, splash: Option<&Url>, message: &str) {
    match splash {
        Some(url) => {
            let mut url = url.clone();
            url.query_pairs_mut().clear().append_pair("error", message);
            let _ = win.navigate(url);
            let _ = win.show();
        }
        None => show_error(win, message),
    }
}

/// Stop the engine gracefully: close its stdin (the lifeline: it writes the running job
/// `interrupted` and exits), wait up to QUIT_WAIT for it, then kill it. Never longer.
fn stop_engine(mut child: Child) {
    drop(child.stdin.take());
    let deadline = Instant::now() + QUIT_WAIT;
    while Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(_)) | Err(_) => return,
            Ok(None) => std::thread::sleep(QUIT_POLL),
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Read the engine's stdout until it ends: point the window at the UI once the engine says it is
/// ready (with `open`, MAATA_OPEN's link, for the first engine only), then keep draining it (closing
/// the pipe would turn every later write in the engine or its libraries into EPIPE). Returns whether
/// it became ready.
fn follow(stdout: ChildStdout, win: &WebviewWindow, token: &str, open: Option<&str>) -> bool {
    let mut ready = false;
    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        if ready {
            println!("{line}");
        } else if let Some(rest) = line.strip_prefix("MAATA_ENGINE_READY ") {
            ready = true;
            let port = rest.split_whitespace().find_map(|kv| kv.strip_prefix("port=")).unwrap_or("0");
            match Url::parse(&format!("http://127.0.0.1:{port}/?token={token}")) {
                Ok(mut url) => {
                    // MAATA_OPEN=<YouTube URL> opens New dub with that link at launch (scripts/verify-mac.sh).
                    if let Some(open) = open {
                        url.query_pairs_mut().append_pair("v", open);
                    }
                    let _ = win.navigate(url);
                }
                Err(e) => show_error(win, &e.to_string()),
            }
        }
    }
    ready
}

/// Run the engine for the app's life: start it, and start it again if it dies on its own (§5).
fn run_engine(handle: tauri::AppHandle, win: WebviewWindow, splash: Option<Url>) {
    let state = handle.state::<Engine>();
    let mut restarts: VecDeque<Instant> = VecDeque::new();
    let mut open = std::env::var("MAATA_OPEN").ok();
    let mut first = true;
    loop {
        let token = random_token();
        let mut child = match engine_command(&handle, &token).and_then(|mut c| c.spawn().map_err(|e| format!("Couldn't start the engine: {e}"))) {
            Ok(c) => c,
            Err(e) if first => return show_error(&win, &e),
            Err(e) => return show_error_page(&win, splash.as_ref(), &e),
        };
        let stdout = child.stdout.take();
        *state.child.lock().unwrap() = Some(child);
        if state.quitting.load(Ordering::SeqCst) {
            // Maata quit while it was starting, after the exit hook looked for it.
            if let Some(c) = state.child.lock().unwrap().take() {
                stop_engine(c);
            }
            return;
        }
        let ready = match stdout {
            Some(out) => follow(out, &win, &token, open.take().as_deref()),
            None => false,
        };
        // Its stdout ended: the engine stopped. Quitting stops it on purpose (the exit hook has it).
        if state.quitting.load(Ordering::SeqCst) {
            return;
        }
        if let Some(c) = state.child.lock().unwrap().take() {
            stop_engine(c); // (it has exited, or is about to: this reaps it)
        }
        if first && !ready {
            return show_error(&win, "The engine stopped while loading. Its log is in the terminal that launched Maata.");
        }
        first = false;
        let now = Instant::now();
        while restarts.front().is_some_and(|t| now.duration_since(*t) > RESTART_WINDOW) {
            restarts.pop_front();
        }
        if restarts.len() >= RESTARTS {
            return show_error_page(
                &win,
                splash.as_ref(),
                "The Maata engine stopped 4 times in 10 minutes, so it isn't started again. Quit Maata and open it again; \
                 its log is in ~/Library/Logs/Maata/engine.log.",
            );
        }
        restarts.push_back(now);
        eprintln!("maata: the engine stopped on its own; starting it again ({} of {RESTARTS} in 10 minutes)", restarts.len());
    }
}

fn main() {
    let app = tauri::Builder::default()
        .manage(Engine { child: Mutex::new(None), quitting: AtomicBool::new(false) })
        .setup(|app| {
            let builder = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("splash.html".into()))
                .title("Maata")
                .inner_size(1440.0, 900.0)
                .min_inner_size(960.0, 640.0);
            #[cfg(target_os = "macos")]
            let builder = builder.title_bar_style(tauri::TitleBarStyle::Overlay).hidden_title(true);
            let win = builder.build()?;
            let splash = win.url().ok();
            let handle = app.handle().clone();
            std::thread::spawn(move || run_engine(handle, win, splash));
            Ok(())
        })
        .on_window_event(|window, event| {
            // macOS: closing the window hides it; the engine and its jobs go on (the Dock icon shows it
            // again). Windows and Linux keep close-quits.
            #[cfg(target_os = "macos")]
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                if window.label() == "main" {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
            #[cfg(not(target_os = "macos"))]
            let _ = (window, event);
        })
        .build(tauri::generate_context!())
        .expect("error while building Maata");

    app.run(|handle, event| match event {
        RunEvent::ExitRequested { .. } => {
            handle.state::<Engine>().quitting.store(true, Ordering::SeqCst);
        }
        // ⌘Q, the Dock's Quit, a logout: never blocked or asked about; the engine gets QUIT_WAIT.
        RunEvent::Exit => {
            let state = handle.state::<Engine>();
            state.quitting.store(true, Ordering::SeqCst);
            let child = state.child.lock().unwrap().take();
            if let Some(child) = child {
                stop_engine(child);
            }
        }
        #[cfg(target_os = "macos")]
        RunEvent::Reopen { .. } => {
            if let Some(win) = handle.get_webview_window("main") {
                let _ = win.show();
                let _ = win.set_focus();
            }
        }
        _ => {}
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    struct RuntimeFixture {
        root: PathBuf,
        resources: PathBuf,
        data: PathBuf,
    }

    impl RuntimeFixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!("maata-runtime-test-{}", random_token()));
            let resources = root.join("resources");
            let data = root.join("application-support");
            fs::create_dir_all(resources.join("runtime")).unwrap();
            fs::create_dir_all(data.join("engine/.venv/bin")).unwrap();
            Self { root, resources, data }
        }

        fn bundle(&self, id: &str) {
            fs::write(self.resources.join("runtime/release.json"),
                serde_json::json!({"schema": 1, "runtime_id": id}).to_string()).unwrap();
        }

        fn install(&self, id: &str) {
            fs::write(self.data.join("engine/installed-runtime.json"),
                serde_json::json!({"runtime_id": id, "setup_version": 1}).to_string()).unwrap();
            fs::write(self.data.join("engine/.venv/bin/maata-engine"), b"fixture launcher, never executed").unwrap();
        }
    }

    impl Drop for RuntimeFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn runtime_ids_require_a_full_lowercase_hash_and_a_supported_bundle_schema() {
        let id = "a1".repeat(32);
        assert_eq!(runtime_id(serde_json::json!({"schema": 1, "runtime_id": id}).to_string().as_bytes(), true), Ok(id.clone()));
        for invalid in ["", "../engine", &"a".repeat(63), &"A".repeat(64), &"g".repeat(64)] {
            assert!(runtime_id(serde_json::json!({"schema": 1, "runtime_id": invalid}).to_string().as_bytes(), true).is_err());
        }
        for invalid in [b"not json".as_slice(), b"[]", b"{}", b"{\"runtime_id\":3}"] {
            assert!(runtime_id(invalid, false).is_err());
        }
        assert!(runtime_id(serde_json::json!({"schema": 2, "runtime_id": id}).to_string().as_bytes(), true).is_err());
        assert!(runtime_id(serde_json::json!({"runtime_id": id}).to_string().as_bytes(), true).is_err());
    }

    #[test]
    fn release_refuses_old_launchers_without_a_matching_installed_stamp() {
        let fixture = RuntimeFixture::new();
        fixture.bundle(&"a".repeat(64));
        fs::write(fixture.data.join("engine/.venv/bin/maata-engine"), b"old checkout shim").unwrap();
        let error = installed_engine(&fixture.resources, &fixture.data).unwrap_err();
        assert!(error.contains("not installed"));
        assert!(error.contains("Setup Maata.command"));
        fixture.install(&"b".repeat(64));
        assert!(installed_engine(&fixture.resources, &fixture.data).unwrap_err().contains("different Maata release"));
        fs::write(fixture.data.join("engine/installed-runtime.json"), b"corrupt").unwrap();
        assert!(installed_engine(&fixture.resources, &fixture.data).unwrap_err().contains("not installed"));
    }

    #[test]
    fn matching_release_requires_both_its_manifest_and_launcher() {
        let fixture = RuntimeFixture::new();
        fixture.install(&"c".repeat(64));
        assert!(installed_engine(&fixture.resources, &fixture.data).unwrap_err().contains("bundled runtime information"));
        fixture.bundle(&"c".repeat(64));
        let launcher = fixture.data.join("engine/.venv/bin/maata-engine");
        assert_eq!(installed_engine(&fixture.resources, &fixture.data).unwrap(), launcher);
        fs::remove_file(&launcher).unwrap();
        assert!(installed_engine(&fixture.resources, &fixture.data).unwrap_err().contains("launcher is missing"));
        fs::create_dir(&launcher).unwrap();
        assert!(installed_engine(&fixture.resources, &fixture.data).is_err());
    }

    #[test]
    fn dock_tool_paths_are_predictable_and_preserve_custom_locations_afterward() {
        let path = native_path(Path::new("/app data"), Path::new("/home/me"),
            Some(OsStr::new("/custom/codex/bin:/opt/homebrew/bin:/another/bin"))).unwrap();
        let paths: Vec<PathBuf> = std::env::split_paths(&path).collect();
        assert_eq!(paths[0], PathBuf::from("/app data/engine/.venv/bin"));
        assert_eq!(paths[1], PathBuf::from("/app data/bin"));
        assert_eq!(paths[2], PathBuf::from("/home/me/.local/bin"));
        assert_eq!(paths[3], PathBuf::from("/home/me/.deno/bin"));
        assert_eq!(paths.iter().filter(|p| **p == PathBuf::from("/opt/homebrew/bin")).count(), 1);
        assert_eq!(&paths[paths.len() - 2..], &[PathBuf::from("/custom/codex/bin"), PathBuf::from("/another/bin")]);
    }
}
