"""Exercise the Finder setup launcher without installing or touching /Applications."""
import os
from pathlib import Path
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def launcher(tmp_path):
    volume = tmp_path / "Mounted installer with spaces"
    volume.mkdir()
    fallback = tmp_path / "Applications fixture" / "Maata.app/Contents/Resources/runtime"
    command = volume / "Setup Maata.command"
    source = (ROOT / "scripts/Setup Maata.command").read_text()
    # Only redirect the system installation path; execute the actual launcher logic.
    assignment = 'maata_payload="/Applications/Maata.app/Contents/Resources/runtime"'
    assert source.count(assignment) == 1
    command.write_text(source.replace(assignment, f"maata_payload={shlex.quote(str(fallback))}"))
    return command, volume / "Maata.app/Contents/Resources/runtime", fallback, tmp_path / "calls"


def payload(path, *, status=0, installer=True):
    path.mkdir(parents=True)
    (path / "release.json").write_text('{"schema":1}')
    if installer:
        script = path / "scripts/install-runtime.sh"
        script.parent.mkdir()
        script.write_text(
            '#!/usr/bin/env bash\nset -eu\n'
            'printf "%s\\n" "$0" "$#" "$1" >> "$MAATA_SETUP_TEST_CALLS"\n'
            + ('echo "Fixture prerequisite is missing" >&2\n' if status else '')
            + f"exit {status}\n"
        )


def run_launcher(fixture):
    command, _, _, calls = fixture
    env = dict(os.environ, MAATA_SETUP_TEST_CALLS=str(calls))
    # Keep stdin open with no input: an accidental non-TTY Enter prompt would block.
    with subprocess.Popen(
        ["bash", str(command)], cwd=command.parent.parent, env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) as process:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            pytest.fail("Setup waited for input in a non-interactive session")
        stdout, stderr = process.communicate()
        assert "Press Return" not in stdout + stderr
        return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def assert_called(fixture, selected):
    assert fixture[3].read_text().splitlines() == [
        str(selected / "scripts/install-runtime.sh"), "1", str(selected),
    ]


def test_sibling_app_wins_over_installed_app_and_paths_with_spaces_are_preserved(launcher):
    _, sibling, fallback, _ = launcher
    payload(sibling)
    payload(fallback, status=42)
    result = run_launcher(launcher)
    assert result.returncode == 0, result.stderr
    assert_called(launcher, sibling)


@pytest.mark.parametrize("sibling_directory", [False, True])
def test_installed_app_fallback_when_sibling_has_no_release_payload(launcher, sibling_directory):
    _, sibling, fallback, _ = launcher
    if sibling_directory:
        sibling.mkdir(parents=True)
    payload(fallback)
    result = run_launcher(launcher)
    assert result.returncode == 0, result.stderr
    assert_called(launcher, fallback)


def test_incomplete_sibling_payload_does_not_silently_install_another_release(launcher):
    _, sibling, fallback, calls = launcher
    payload(sibling, installer=False)
    payload(fallback)
    result = run_launcher(launcher)
    assert result.returncode == 1
    assert "Cannot find the Maata runtime" in result.stderr
    assert not calls.exists()


def test_missing_payload_reports_actionable_error_without_running_installer(launcher):
    result = run_launcher(launcher)
    assert result.returncode == 1
    assert "Cannot find the Maata runtime" in result.stderr
    assert "beside Maata.app" in result.stderr
    assert "setup did not finish" in result.stderr
    assert not launcher[3].exists()


def test_installer_failure_preserves_exit_code_and_original_error_without_fallback(launcher):
    _, sibling, fallback, _ = launcher
    payload(sibling, status=37)
    payload(fallback)
    result = run_launcher(launcher)
    assert result.returncode == 37
    assert "Fixture prerequisite is missing" in result.stderr
    assert "setup did not finish" in result.stderr
    assert_called(launcher, sibling)
