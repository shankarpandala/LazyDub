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
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
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
