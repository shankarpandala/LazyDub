"""Installer control flow with fake provisioning; never installs packages or models."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/install-runtime.sh"


@pytest.fixture
def installer():
    # Exercise the exact stdlib program embedded in the distributable shell file.
    body = SCRIPT.read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    body = body.rsplit("\ntry:\n    main()", 1)[0]
    namespace = {}
    exec(compile(body, str(SCRIPT), "exec"), namespace)
    return namespace


def payload(tmp_path, suffix=""):
    root = tmp_path / ("payload" + suffix)
    names = ["engine/pyproject.toml", "engine/uv.lock", "engine/models.lock.json",
             "engine/runtimes/omnivoice/pyproject.toml", "engine/runtimes/omnivoice/uv.lock",
             "engine/runtimes/omnivoice/models.lock.json", "scripts/setup-omnivoice.sh",
             "scripts/install-runtime.sh", "engine/src/maata_engine/server.py", "LICENSE"]
    files = {}
    for name in names:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        content = SCRIPT.read_bytes() if name == "scripts/install-runtime.sh" else (name + suffix).encode()
        p.write_bytes(content)
        files[name] = hashlib.sha256(content).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    release = dict(schema=1, runtime_id=digest, payload_sha256=digest, files=files,
                   app_version="0.1.0", platform="macos-arm64")
    (root / "release.json").write_text(json.dumps(release))
    return root, release


@pytest.fixture
def provision(installer, tmp_path, monkeypatch):
    install = tmp_path / "Application Support" / "Maata install"
    data = tmp_path / "Application Support" / "Maata data"
    monkeypatch.setenv("MAATA_INSTALL_ROOT", str(install))
    monkeypatch.setenv("MAATA_DATA_DIR", str(data))
    monkeypatch.delenv("MAATA_SETUP_OFFLINE", raising=False)
    monkeypatch.setattr(installer["platform"], "system", lambda: "Darwin")
    monkeypatch.setattr(installer["platform"], "machine", lambda: "arm64")
    monkeypatch.setattr(installer["shutil"], "which", lambda name: sys.executable if name == "codex" else "/fake/" + name)
    monkeypatch.setattr(installer["shutil"], "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    installer["no_engine"] = lambda: None
    calls = []
    def fake_run(command, *, env=None, capture=False):
        args = list(map(str, command))
        calls.append((args, env))
        if "sync" in args:
            bindir = Path(env["UV_PROJECT_ENVIRONMENT"]) / "bin"
            bindir.mkdir(parents=True, exist_ok=True)
            for name in ("python", "maata-engine", "maata-bench"):
                p = bindir / name
                p.write_text("#!/bin/sh\nexit 0\n")
                p.chmod(0o755)
        if args[0] == "/bin/bash":
            p = data / "runtimes/omnivoice/bin/python"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("#!/bin/sh\nexit 0\n")
            p.chmod(0o755)
        return "" if capture else None
    installer["run"] = fake_run
    def invoke(source, *flags):
        monkeypatch.setattr(sys, "argv", [str(SCRIPT), *flags, str(source)])
        installer["main"]()
    return SimpleNamespace(install=install, data=data, calls=calls, invoke=invoke, run=fake_run)


@pytest.mark.parametrize("mode", ["--check", "--dry-run"])
def test_check_is_read_only(installer, provision, tmp_path, mode):
    source, _ = payload(tmp_path)
    provision.invoke(source, mode)
    assert not provision.install.exists()
    assert not provision.data.exists()
    assert provision.calls == []


@pytest.mark.parametrize("corruption", ["contents", "extra", "symlink", "digest", "traversal"])
def test_payload_fails_closed(installer, tmp_path, corruption):
    source, release = payload(tmp_path)
    if corruption == "contents":
        (source / "engine/uv.lock").write_text("tampered")
    elif corruption == "extra":
        (source / "private.txt").write_text("unexpected")
    elif corruption == "symlink":
        p = source / "engine/uv.lock"
        p.unlink()
        p.symlink_to(source / "engine/pyproject.toml")
    elif corruption == "digest":
        release["runtime_id"] = "0" * 64
        (source / "release.json").write_text(json.dumps(release))
    else:
        release["files"]["../outside"] = "0" * 64
        digest = hashlib.sha256(json.dumps(release["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        release.update(runtime_id=digest, payload_sha256=digest)
        (source / "release.json").write_text(json.dumps(release))
    with pytest.raises(RuntimeError):
        installer["verify_payload"](source)


@pytest.mark.parametrize(("executable", "args"), [
    ("/Users/me/Library/Application Support/Maata/engine/.venv/bin/python3.12",
     "/Users/me/Library/Application Support/Maata/engine/.venv/bin/python3.12 /Users/me/Library/Application Support/Maata/engine/.venv/bin/maata-engine --port 1"),
    ("/some path/.venv/bin/maata-engine", "/some path/.venv/bin/maata-engine --port 1"),
    ("/usr/local/bin/python3", "/usr/local/bin/python3 -m maata_engine.server --port 1"),
    ("/opt/homebrew/bin/uv", "uv run --no-sync maata-engine --port 1"),
])
def test_running_engine_with_spaces_refuses(installer, monkeypatch, executable, args):
    def fake_ps(command, **kwargs):
        return SimpleNamespace(stdout="123 " + (executable if command[-1] == "pid=,comm=" else args) + "\n")
    monkeypatch.setattr(installer["subprocess"], "run", fake_ps)
    with pytest.raises(RuntimeError, match="active engine PID 123"):
        installer["no_engine"]()


def test_unrelated_search_and_installer_are_not_engines(installer, monkeypatch):
    comm = "12 /bin/zsh\n13 /bin/rg\n14 /local/bin/python3\n15 /local/bin/python3\n"
    args = "12 /bin/zsh scripts/install-runtime.sh payload\n13 rg maata-engine\n14 python3 - payload\n15 python3 -c print('maata-engine')\n"
    monkeypatch.setattr(installer["subprocess"], "run", lambda command, **_: SimpleNamespace(stdout=comm if command[-1] == "pid=,comm=" else args))
    installer["no_engine"]()


def test_success_uses_final_paths_and_independent_copy(installer, provision, tmp_path):
    source, release = payload(tmp_path)
    provision.invoke(source)
    version = provision.install / "runtimes" / release["runtime_id"]
    stable = provision.install / "engine"
    assert stable.is_symlink() and stable.resolve().parent == version
    assert json.loads((stable / "installed-runtime.json").read_text())["health"] == "passed"
    assert (version / "engine/src/maata_engine/server.py").read_bytes() == (source / "engine/src/maata_engine/server.py").read_bytes()
    assert not any(p.is_symlink() for p in version.rglob("*") if "activation-" not in str(p))
    sync, env = provision.calls[0]
    assert sync == ["/fake/uv", "sync", "--project", str(version / "engine"), "--frozen", "--no-dev", "--extra", "apple", "--extra", "resolve", "--python", "3.12"]
    assert env["UV_PROJECT_ENVIRONMENT"] == str(version / "engine/.venv")
    assert "PYTHONPATH" not in env
    setup = next(c for c in provision.calls if c[0][0] == "/bin/bash")
    assert setup[1]["MAATA_ENGINE_PYTHON"] == str(version / "engine/.venv/bin/python")
    launcher = (stable / ".venv/bin/maata-engine").read_text()
    assert "$HOME/.deno/bin" in launcher and str(version) in launcher
    assert " --models '" in launcher and ' "$@"' in launcher
    assert not (provision.install / ".runtime-install.lock").exists()


@pytest.mark.parametrize("failure", ["sync", "health", "runtime_health"])
def test_failure_preserves_legacy_launcher_and_user_work(installer, provision, tmp_path, failure):
    source, _ = payload(tmp_path)
    old = provision.install / "engine"
    old.mkdir(parents=True)
    (old / "original").write_text("old launcher")
    media = provision.data / "saved.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"unchanged video")
    def failing_run(command, **kwargs):
        args = list(map(str, command))
        if ((failure == "sync" and "sync" in args)
                or (failure == "health" and any("Required pinned models" in s for s in args))
                or (failure == "runtime_health" and "-I" in args)):
            raise RuntimeError("deliberate provision failure")
        return provision.run(command, **kwargs)
    installer["run"] = failing_run
    with pytest.raises(RuntimeError, match="deliberate"):
        provision.invoke(source)
    assert not old.is_symlink()
    assert (old / "original").read_text() == "old launcher"
    assert media.read_bytes() == b"unchanged video"
    assert not (provision.install / ".runtime-install.lock").exists()


def test_legacy_migration_and_later_upgrade_retain_old_versions(installer, provision, tmp_path):
    old = provision.install / "engine"
    old.mkdir(parents=True)
    (old / "original").write_text("old launcher")
    source, first = payload(tmp_path)
    provision.invoke(source)
    first_activation = old.resolve()
    backups = list(provision.install.glob("engine-before-install-*"))
    assert len(backups) == 1 and (backups[0] / "original").read_text() == "old launcher"
    source2, second = payload(tmp_path, "-new")
    provision.invoke(source2)
    assert old.resolve() != first_activation and first_activation.exists()
    assert json.loads((old / "installed-runtime.json").read_text())["runtime_id"] == second["runtime_id"]
    assert (provision.install / "runtimes" / first["runtime_id"] / "engine/uv.lock").exists()


def test_legacy_promotion_failure_rolls_back(installer, tmp_path, monkeypatch):
    install = tmp_path / "install"
    stable = install / "engine"
    stable.mkdir(parents=True)
    (stable / "original").write_text("old")
    activation = tmp_path / "new"
    activation.mkdir()
    monkeypatch.setattr(installer["os"], "replace", lambda *_: (_ for _ in ()).throw(OSError("promotion failed")))
    with pytest.raises(OSError):
        installer["promote"](install, activation)
    assert not stable.is_symlink() and (stable / "original").read_text() == "old"
    assert not list(install.glob(".engine-link-*"))


def test_existing_source_changes_are_never_overwritten(installer, provision, tmp_path):
    source, release = payload(tmp_path)
    provision.invoke(source)
    target = provision.install / "runtimes" / release["runtime_id"] / "engine/uv.lock"
    target.write_text("changed local source")
    current = (provision.install / "engine").resolve()
    with pytest.raises(RuntimeError, match="changed source"):
        provision.invoke(source)
    assert target.read_text() == "changed local source"
    assert (provision.install / "engine").resolve() == current


def test_headroom_first_install_but_not_existing_model_repair(installer, tmp_path, monkeypatch):
    monkeypatch.setattr(installer["shutil"], "disk_usage", lambda _: SimpleNamespace(free=1024))
    with pytest.raises(RuntimeError, match="20 GiB"):
        installer["check_headroom"](tmp_path / "install", tmp_path / "data", tmp_path / "version")
    cached = tmp_path / "data/Models/model/file"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached")
    installer["check_headroom"](tmp_path / "install", tmp_path / "data", tmp_path / "version")


def test_offline_sync_and_no_main_fetch(installer, provision, tmp_path, monkeypatch):
    source, _ = payload(tmp_path)
    monkeypatch.setenv("MAATA_SETUP_OFFLINE", "1")
    provision.invoke(source)
    assert "--offline" in provision.calls[0][0]
    assert all(env["UV_OFFLINE"] == "1" for _, env in provision.calls)
    assert not any("fetch" in args for args, _ in provision.calls)


def test_offline_subprocess_tree_has_network_denied(installer, monkeypatch):
    calls = []
    monkeypatch.setattr(installer["subprocess"], "run", lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0, stdout=""))
    installer["run"](["/bin/bash", "setup-omnivoice.sh"], env={"UV_OFFLINE": "1"})
    assert calls == [["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)", "/bin/bash", "setup-omnivoice.sh"]]
