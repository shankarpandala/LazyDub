#!/usr/bin/env bash
# Finder opens .command files in Terminal. Setup never launches Maata automatically.
set -euo pipefail

maata_setup_finish() {
  maata_setup_status=$?
  trap - EXIT
  if [[ "$maata_setup_status" -ne 0 ]]; then
    printf '\nMaata setup did not finish. Read the error above, correct it, and run setup again.\n' >&2
  fi
  if [[ -t 0 && -t 1 ]]; then
    printf '\nPress Return to close this setup session. '
    read -r _ || true
  fi
  exit "$maata_setup_status"
}
trap maata_setup_finish EXIT

maata_setup_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
maata_payload="$maata_setup_dir/Maata.app/Contents/Resources/runtime"
if [[ ! -f "$maata_payload/release.json" ]]; then
  maata_payload="/Applications/Maata.app/Contents/Resources/runtime"
fi
if [[ ! -f "$maata_payload/release.json" || ! -f "$maata_payload/scripts/install-runtime.sh" ]]; then
  echo "Cannot find the Maata runtime. Keep Setup Maata.command beside Maata.app in the installer, or copy Maata.app to /Applications first." >&2
  exit 1
fi

printf 'Quit Maata before setup. Installing the runtime bundled with:\n%s\n\n' "$maata_payload"
bash "$maata_payload/scripts/install-runtime.sh" "$maata_payload"
