# Install the macOS release

This guide covers the Apple Silicon DMG. Building Maata from source is a separate developer workflow in the [README](../README.md#build-from-source).

## Requirements

- An Apple Silicon Mac (arm64), running macOS 14 or later. This DMG does not support Intel Macs.
- At least 16 GB of memory; 24 GB is recommended.
- At least 20 GiB (about 21.5 GB) free before setup; **30 GB or more is recommended**, with additional space for downloaded videos, temporary audio and exported MP4s.
- An internet connection for first-time setup, YouTube downloads and text translation.
- [uv](https://docs.astral.sh/uv/getting-started/installation/), [Deno](https://docs.deno.com/runtime/getting_started/installation/) and [Codex CLI](https://developers.openai.com/codex/cli/), plus Python 3.12 installed through uv. Node.js and Rust are not end-user prerequisites for the DMG.
- Your own Codex sign-in, with access to Maata's configured text model (`gpt-6-luna`). Setup does not create an account, sign in, purchase credits or change your plan.

If you already use Homebrew, the prerequisites can be installed in Terminal:

```bash
brew install uv deno
brew install --cask codex
uv python install 3.12
codex login
```

Alternatively, use the linked official installers. Sign-in is interactive and stays under your control. Maata sends transcript, translation and video-context text through Codex; audio inference stays local.

## Install and run setup

1. Download the Apple Silicon DMG from the project's [GitHub Releases](https://github.com/shankarpandala/LazyDub/releases). Review its release notes and checksum when supplied.
2. Open the DMG. It contains **Maata.app**, **Setup Maata.command**, and an **Applications** shortcut.
3. Drag **Maata.app** onto **Applications**. Run the installed copy from Applications, rather than leaving your only copy on the mounted image.
4. **Quit Maata before setup.** Double-click **Setup Maata.command** in the mounted DMG. It opens Terminal and reports prerequisites, downloads and verification. It finds the adjacent Maata.app, or the installed copy at `/Applications/Maata.app`. Keep the DMG mounted until it finishes.
5. Follow any reported prerequisite or authentication instructions. If setup stops, correct the reported problem and run it again; verified model files can be reused.
6. After setup reports success, open **Maata** from Applications. Check Settings for the Codex sign-in and model status, then add a video.

The app and engine are separate installations. Copying the app alone does not install the Python environments or model weights. Setup installs an independent copy of the engine source and its pinned runtime under Application Support; it does not depend on a developer checkout remaining on your disk. OmniVoice has its own pinned environment so it does not replace the main engine's Python dependencies.

The current Apple model manifests contain approximately **8.58 GB** of files, including the legacy Chatterbox rollback model. The download is separate from the DMG. Python environments, package caches and temporary download files require additional space. The 20 GiB (about 21.5 GB) allowance is a minimum, not a bound on a long video's total storage use.

## macOS security prompts

This release is **ad-hoc signed, not Developer ID signed or notarized by Apple**. macOS may refuse to open the app or setup command initially. Only proceed after reviewing the source/release and deciding you trust the download.

After trying to open the blocked item, go to **System Settings → Privacy & Security**, find its blocked-app notice and choose **Open Anyway** if appropriate. Confirm the subsequent prompt. This is Apple's documented per-app approval process; see [Safely open apps on your Mac](https://support.apple.com/en-us/102445). If macOS reports actual malware or a damaged/modified download, stop and check the release rather than overriding that warning.

## Optional multi-speaker model

Speaker diarization uses the gated [pyannote speaker-diarization-community-1 model](https://huggingface.co/pyannote/speaker-diarization-community-1). To download it, first accept its terms using your Hugging Face account, then provide an authorized `HF_TOKEN` environment variable when running setup from Terminal. Finder-launched commands do not inherit variables set in another Terminal window.

For example, after setting `HF_TOKEN` in your current Terminal, drag **Setup Maata.command** from the DMG into that Terminal window and press Return. Do not include tokens in bug reports or commit them to source control. Setup does not accept model terms or log in for you.

Without that model, Maata can use a single-speaker fallback; it cannot reliably assign different profiles to multiple people in the same video. Add the model before starting a multi-speaker dub, then re-run setup to verify it is available.

## Select the output voices

The Apple engine uses OmniVoice Male/Female synthesis profiles, not clones of English source voices. Each speaker's selector offers **Auto**, **Male** and **Female**.

Auto uses bounded pitch evidence from clean source speech. This estimates a suitable vocal range, not the person's gender identity. When the range is ambiguous or there is too little usable speech, Maata asks you to choose Male or Female and resume. A manual selection overrides the acoustic estimate. Each resolved profile has its own calibration and audio-cache identity.

A profile is not a promise of identical voice character across sentences or perfect pronunciation. Listen to the result; representative Telugu listening and long-video quality checks are still ongoing.

## Files and storage

The application identifier directory and the model/settings directory are intentionally separate:

| Location | Contents |
| --- | --- |
| `/Applications/Maata.app` | Desktop shell, UI and setup resources |
| `~/Library/Application Support/io.github.shankarpandala.maata/engine/` | Pointer to the active engine launcher and installation record |
| `~/Library/Application Support/io.github.shankarpandala.maata/runtimes/<runtime-id>/engine/` | Versioned main engine source and `.venv` runtime |
| `~/Library/Application Support/Maata/runtimes/omnivoice/` | Isolated OmniVoice Python runtime |
| `~/Library/Application Support/Maata/Models/` | Verified local model files |
| `~/Library/Application Support/Maata/settings.json` | Engine settings, including the output folder |
| `~/Library/Caches/Maata/<video id>/` | Downloaded media, transcript/text caches and resumable job work |
| `~/Movies/Maata/` | Default output folder for dubbed MP4s |

Job traces can contain source and translated text. Review them before sharing. Package-manager caches may live outside these directories and consume additional space.

## Updating or troubleshooting

- **Update:** quit Maata, replace the app in Applications with the new release, then run that release's setup command before launching it. Keep completed outputs and job data; an app update is not a request to restart cancelled dubs.
- **Engine not installed:** copying the app was only the first step. Run the matching setup command and read its final result.
- **A prerequisite is missing:** install the named tool using its official instructions, open a new Terminal if needed, and run setup again.
- **Codex is not signed in or access is unavailable:** run `codex login` yourself and inspect Settings. Maata does not silently switch translation providers or accounts.
- **Model download or verification failed:** check connectivity, available disk space and, for pyannote, accepted terms and token access. Re-run setup; do not substitute unverified model files.
- **Source voice profile is unresolved:** choose Male or Female for the indicated speaker, then resume.
- **More disk space is needed:** allow for the current video's temporary media and exported MP4 in addition to the installed models. Avoid deleting a running job's cache.

Closing the window leaves background work running. Quitting stops the engine gracefully; explicitly paused or cancelled jobs remain stopped until you resume them.
