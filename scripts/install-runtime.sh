#!/usr/bin/env bash
# Install a verified release payload independently of the checkout. --check and
# --dry-run inspect only; neither creates environments nor downloads anything.
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$HOME/.deno/bin:$PATH"
maata_install_python="${MAATA_INSTALL_PYTHON:-}"
if [[ -z "$maata_install_python" ]] && command -v uv >/dev/null; then
  maata_install_python="$(uv python find --no-project --no-python-downloads 3.12 2>/dev/null || true)"
fi
if [[ -z "$maata_install_python" ]]; then
  maata_install_python="$(command -v python3 || true)"
  # Apple's stub can open an Xcode installation dialog on a clean Mac.
  [[ "$maata_install_python" != /usr/bin/python3 ]] || maata_install_python=""
fi
[[ -n "$maata_install_python" && -x "$maata_install_python" ]] || {
  echo "A bootstrap Python 3 is needed. Run: uv python install 3.12, then retry." >&2; exit 1;
}
exec "$maata_install_python" - "$@" <<'PY'
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid


def fail(message):
    raise RuntimeError(message)


def run(command, *, env=None, capture=False):
    # setup-omnivoice also verifies/fetches models. Offline means no network for
    # that subprocess tree, including a missing model or an unexpected package hook.
    if env and env.get("UV_OFFLINE") == "1":
        command = ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)", *command]
    result = subprocess.run([str(x) for x in command], env=env, text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None)
    if result.returncode:
        fail("Command failed: " + str(command[0]) + ". Earlier runtimes and the launcher have not been removed.")
    return result.stdout.strip() if capture else None


def no_engine():
    # ps does not quote paths containing spaces. Read executable names separately
    # instead of attempting to reconstruct argv with shlex.split.
    names = subprocess.run(["ps", "-axo", "pid=,comm="], capture_output=True, text=True, check=True)
    commands = subprocess.run(["ps", "-axo", "pid=,args="], capture_output=True, text=True, check=True)
    executables = {}
    for row in names.stdout.splitlines():
        parts = row.strip().split(None, 1)
        if len(parts) == 2:
            executables[parts[0]] = Path(parts[1]).name.lower()
    active = []
    for row in commands.stdout.splitlines():
        try:
            pid, command = row.strip().split(None, 1)
        except ValueError:
            continue
        first = executables.get(pid, "")
        python = first.startswith("python")
        engine_argument = re.search(r"(?:^|/|\s)maata-engine(?:\s|$)", command)
        # A Python inline command/test argument mentioning an engine is not an engine.
        inline = re.search(r"\s-(?:c|m\s+(?!maata_engine\.server(?:\s|$)))", command)
        direct = first == "maata-engine" or (python and engine_argument and not inline)
        module = python and re.search(r"\s-m\s+maata_engine\.server(?:\s|$)", command)
        uv = first == "uv" and re.search(r"\srun\s", command) and engine_argument
        if direct or module or uv:
            active.append(pid)
    if active:
        fail("Quit Maata before installing or checking its runtime (active engine PID " + ", ".join(active) + ").")


def check_headroom(install, data, version):
    # Existing installations/model caches need only a repair/upgrade, not another
    # complete weight download. This conservative first-install check is read-only.
    models = data / "Models"
    if version.exists() or (models.is_dir() and any(p.is_file() for p in models.rglob("*"))):
        return
    destination = install
    while not destination.exists():
        destination = destination.parent
    if shutil.disk_usage(destination).free < 20 * 1024 ** 3:
        fail("A first Maata runtime/model installation needs at least 20 GiB of free disk space. Free space and retry.")


def verify_payload(root):
    if root.is_symlink() or not root.is_dir():
        fail("Payload must be a real directory, not a symlink.")
    manifest = root / "release.json"
    if manifest.is_symlink():
        fail("release.json must not be a symlink.")
    release = json.loads(manifest.read_text())
    digest, files = release.get("payload_sha256"), release.get("files")
    if (release.get("schema") != 1 or release.get("platform") != "macos-arm64"
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or release.get("runtime_id") != digest or not isinstance(files, dict)):
        fail("Unsupported release manifest or invalid runtime ID.")
    calculated = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if calculated != digest:
        fail("Release payload digest does not match its file manifest.")
    required = {"engine/pyproject.toml", "engine/uv.lock", "engine/models.lock.json",
                "engine/runtimes/omnivoice/pyproject.toml", "engine/runtimes/omnivoice/uv.lock",
                "engine/runtimes/omnivoice/models.lock.json", "scripts/setup-omnivoice.sh", "scripts/install-runtime.sh"}
    if not required <= files.keys() or not any(p.startswith("engine/src/maata_engine/") for p in files):
        fail("Release payload is missing engine sources, locked dependencies or setup scripts.")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            fail("Symlinks are not allowed in a release payload.")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files) | {"release.json"}:
        fail("Release payload contains missing or unexpected files.")
    for relative, expected in files.items():
        parts = PurePosixPath(relative)
        if (not relative or parts.is_absolute() or ".." in parts.parts or str(parts) != relative
                or relative == "release.json" or not isinstance(expected, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            fail("Invalid path or checksum in release manifest.")
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            fail("Release payload checksum mismatch: " + relative)
    return release


def check_existing_copy(root, release):
    # Installed dependencies and generated bytecode may coexist with the immutable source snapshot.
    for relative, expected in release["files"].items():
        path = root / relative
        linked_parent = any(parent.is_symlink() for parent in path.parents if parent != root and root in parent.parents)
        if linked_parent or path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            fail("Existing version has changed source files; preserving it. Choose a fresh release payload: " + relative)
    if (root / "release.json").is_symlink() or json.loads((root / "release.json").read_text()) != release:
        fail("Existing version manifest differs; preserving it.")


def promote(install, activation):
    stable = install / "engine"
    temporary = install / (".engine-link-" + uuid.uuid4().hex)
    temporary.symlink_to(activation, target_is_directory=True)
    legacy = None
    try:
        if stable.exists() and not stable.is_symlink():
            # The first migration from a physical development shim has a brief fail-closed maintenance gap.
            # Preserve it and roll back on failure. Later symlink upgrades switch launcher+stamp atomically.
            legacy = install / ("engine-before-install-" + time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
            stable.rename(legacy)
        os.replace(temporary, stable)
    except BaseException:
        if legacy is not None and not stable.exists() and not stable.is_symlink():
            legacy.rename(stable)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    if legacy is not None:
        print("Previous engine launcher preserved at:", legacy)


def main():
    parser = argparse.ArgumentParser(description="Install the pinned Maata engine without a development checkout.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify payload and prerequisites without changing files")
    mode.add_argument("--dry-run", action="store_true", help="show verified installation paths without changing files")
    parser.add_argument("payload", type=Path)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        fail("This runtime release requires an Apple Silicon Mac.")
    payload = args.payload.expanduser().absolute()
    release = verify_payload(payload)
    no_engine()
    uv, deno = shutil.which("uv"), shutil.which("deno")
    if not uv:
        fail("Install uv from https://docs.astral.sh/uv/getting-started/installation/ and retry.")
    if not deno:
        fail("Install Deno (for YouTube resolution), for example: brew install deno. Then retry.")
    codex = next((p for p in (os.environ.get("MAATA_CODEX_BIN"), shutil.which("codex"),
                  "/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex",
                  str(Path.home() / ".local/bin/codex")) if p and Path(p).is_file() and os.access(p, os.X_OK)), None)
    if not codex:
        fail("Install the Codex CLI or ChatGPT desktop app, sign in with codex login, then retry. Audio stays local.")
    install = Path(os.environ.get("MAATA_INSTALL_ROOT", str(Path.home() / "Library/Application Support/io.github.shankarpandala.maata"))).expanduser().absolute()
    data = Path(os.environ.get("MAATA_DATA_DIR", str(Path.home() / "Library/Application Support/Maata"))).expanduser().absolute()
    version = install / "runtimes" / release["runtime_id"]
    check_headroom(install, data, version)
    print("Verified runtime:", release["runtime_id"])
    print("Runtime destination:", version)
    print("Shared model/data destination:", data)
    print("Codex must be signed in with ChatGPT (codex login); no model request is made by this installer.")
    if args.check or args.dry_run:
        print("Check complete; no files changed or downloads started.")
        return
    install.mkdir(parents=True, exist_ok=True)
    lock = install / ".runtime-install.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        fail("Another install may be running. Inspect " + str(lock) + " before retrying; it was not removed.")
    try:
        (lock / "pid").write_text(str(os.getpid()))
        if version.is_symlink():
            fail("The version destination must not be a symlink.")
        if version.exists():
            check_existing_copy(version, release)
        else:
            version.parent.mkdir(parents=True, exist_ok=True)
            staging = version.with_name("." + version.name + ".copy-" + uuid.uuid4().hex)
            shutil.copytree(payload, staging)
            verify_payload(staging)
            staging.rename(version)
        engine = version / "engine"
        python = engine / ".venv/bin/python"
        executable = engine / ".venv/bin/maata-engine"
        env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(engine / ".venv"), "MAATA_DATA_DIR": str(data),
               "MAATA_ENGINE_PYTHON": str(python), "PYTHONNOUSERSITE": "1"}
        env.pop("PYTHONPATH", None)
        sync = [uv, "sync", "--project", engine, "--frozen", "--no-dev", "--extra", "apple", "--extra", "resolve", "--python", "3.12"]
        if os.environ.get("MAATA_SETUP_OFFLINE") == "1":
            sync.append("--offline")
            env["UV_OFFLINE"] = "1"
        run(sync, env=env)
        no_engine()
        if os.environ.get("MAATA_SETUP_OFFLINE") != "1":
            run([engine / ".venv/bin/maata-bench", "fetch", "--backend", "apple", "--models", data / "Models"], env=env)
        run(["/bin/bash", version / "scripts/setup-omnivoice.sh"], env=env)
        run([executable, "--help"], env=env, capture=True)  # imports the engine; never starts it or a queue
        health = r'''
import json,sys
from pathlib import Path
from maata_engine.models import models_for,verify,_spec
from maata_engine.backends.omnivoice import OmniVoiceTTS
root=Path(sys.argv[1]); missing=[]; optional=[]
for model in models_for('apple'):
    bad=[f['path'] for f in model['files'] if not verify(root/'Models'/model['id']/f['path'],_spec(f))]
    if bad:
        (optional if model.get('gated') else missing).append(model['id'])
tts=OmniVoiceTTS(root/'Models/omnivoice',runtime_python=root/'runtimes/omnivoice/bin/python')
missing.extend(tts.missing())
if missing: raise SystemExit('Missing or invalid required model/runtime assets: '+', '.join(missing))
if optional: print('Optional gated models unavailable; source speaker separation is limited: '+', '.join(optional))
print('Required pinned models verified; engine imports and OmniVoice runtime paths are ready.')
'''
        run([python, "-c", health, data], env=env)
        # Check the isolated package/dependency pins without loading the model or touching the GPU.
        runtime_health = "import runpy,sys; runpy.run_path(sys.argv[1])['verify_runtime']()"
        run([data / "runtimes/omnivoice/bin/python", "-I", "-c", runtime_health,
             engine / "src/maata_engine/backends/_omnivoice_worker.py"], env=env)
        no_engine()
        activation = version / ("activation-" + uuid.uuid4().hex)
        launcher = activation / ".venv/bin/maata-engine"
        launcher.parent.mkdir(parents=True)
        launcher.write_text("#!/bin/sh\nexport PATH=\"/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$HOME/.deno/bin:$PATH\"\n"
            + "export MAATA_OMNIVOICE_PYTHON=" + shlex.quote(str(data / "runtimes/omnivoice/bin/python")) + "\n"
            + "exec " + shlex.quote(str(executable)) + " --models " + shlex.quote(str(data / "Models"))
            + " --config-dir " + shlex.quote(str(data)) + ' "$@"\n')
        launcher.chmod(0o755)
        stamp = {"schema": 1, "runtime_id": release["runtime_id"], "payload_sha256": release["payload_sha256"],
                 "app_version": release.get("app_version"), "version_path": str(version), "data_dir": str(data),
                 "installed_at": time.time(), "health": "passed"}
        (activation / "installed-runtime.json").write_text(json.dumps(stamp, indent=2) + "\n")
        promote(install, activation)
        print("Installed. Open Maata.app; prior runtime versions and all user work are preserved.")
    finally:
        (lock / "pid").unlink(missing_ok=True)
        lock.rmdir()


try:
    main()
except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
    print("Maata setup: " + str(exc), file=sys.stderr)
    raise SystemExit(1)
PY
