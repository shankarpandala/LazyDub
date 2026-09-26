"""Maata inference engine.

No telemetry (CLAUDE.md hard constraint). Some libraries phone home by default, so switch them off
here, before any of them can be imported by any entry point (server, bench, tests):
- pyannote.audio 4 sends OpenTelemetry spans (model/pipeline init, audio duration, speaker counts)
  to otel.pyannote.ai unless PYANNOTE_METRICS_ENABLED is false when it is first imported.
- huggingface_hub sends usage telemetry unless disabled (the engine also runs it offline).
Assigned unconditionally, so an inherited environment can't turn them back on.
"""

import os

os.environ["PYANNOTE_METRICS_ENABLED"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
