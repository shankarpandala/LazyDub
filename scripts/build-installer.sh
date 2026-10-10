#!/usr/bin/env bash
# Build a locally ad-hoc-signed Apple Silicon installer. This does not install or launch Maata.
set -euo pipefail

maata_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "Build the Maata installer on an Apple Silicon Mac using an arm64 shell." >&2
  exit 1
fi
for maata_tool in npm node cargo codesign ditto hdiutil shasum; do
  command -v "$maata_tool" >/dev/null || {
    echo "Missing build prerequisite: $maata_tool" >&2
    exit 1
  }
done

maata_version="$(node -e '
  const config = require(process.argv[1]);
  if (!/^\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?$/.test(config.version)) {
    throw new Error("Invalid release version");
  }
  process.stdout.write(config.version);
' "$maata_repo/app/src-tauri/tauri.conf.json")"
maata_name="Maata-$maata_version-macos-arm64.dmg"
maata_dist="$maata_repo/dist"
maata_app="$maata_repo/app/src-tauri/target/release/bundle/macos/Maata.app"

[[ -f "$maata_repo/docs/release-installation.md" ]] || {
  echo "Missing docs/release-installation.md." >&2
  exit 1
}
[[ -x "$maata_repo/scripts/Setup Maata.command" ]] || {
  echo "Missing executable scripts/Setup Maata.command." >&2
  exit 1
}

# Keep the bundle location predictable even when the caller uses a shared Cargo target directory.
(
  cd "$maata_repo/app"
  export CARGO_TARGET_DIR="$maata_repo/app/src-tauri/target"
  npm run tauri build -- --bundles app
)
[[ -f "$maata_app/Contents/Resources/runtime/release.json" ]] || {
  echo "The app is missing its bundled runtime manifest." >&2
  exit 1
}
codesign --verify --deep --strict "$maata_app"

mkdir -p "$maata_dist"
maata_work="$(mktemp -d "$maata_dist/.maata-installer.XXXXXX")"
trap 'rm -rf -- "$maata_work"' EXIT
maata_stage="$maata_work/stage"
mkdir "$maata_stage"
ditto "$maata_app" "$maata_stage/Maata.app"
ln -s /Applications "$maata_stage/Applications"
cp "$maata_repo/scripts/Setup Maata.command" "$maata_stage/Setup Maata.command"
chmod 755 "$maata_stage/Setup Maata.command"
cp "$maata_repo/docs/release-installation.md" "$maata_stage/README.md"

hdiutil create -volname "Maata $maata_version" -srcfolder "$maata_stage" \
  -fs HFS+ -format UDZO "$maata_work/$maata_name"
hdiutil verify "$maata_work/$maata_name"
(
  cd "$maata_work"
  shasum -a 256 "$maata_name" > SHA256SUMS
)

# Do not replace a previous installer until the new image and checksum are complete.
[[ ! -d "$maata_dist/$maata_name" && ! -d "$maata_dist/SHA256SUMS" ]] || {
  echo "Installer output paths must be files, not directories." >&2
  exit 1
}
mv -f "$maata_work/$maata_name" "$maata_dist/$maata_name"
mv -f "$maata_work/SHA256SUMS" "$maata_dist/SHA256SUMS"
printf 'Installer: %s\nChecksum: %s\n' "$maata_dist/$maata_name" "$maata_dist/SHA256SUMS"
