"""Recreate the selected original Voice B reference, entirely offline.

Invoked by setup-omnivoice.sh in this isolated runtime under network denial.
No source-video audio or downloaded reference is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

TEXT = "నమస్కారం! మీరు ఎలా ఉన్నారు? నేను బాగున్నాను."
EXPECTED_SHA256 = "9e6ab4e1566c10ef4ee7984a8e2d7ff072d0f69b0a2a09de245dcecc895752e8"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import numpy as np
    import soundfile as sf
    import torch
    import omnivoice.models.omnivoice as impl

    if not torch.backends.mps.is_available():
        raise RuntimeError("The selected Voice B seed requires Apple MPS")
    def local_only(value):
        path = Path(value)
        if not path.is_dir():
            raise RuntimeError("Automatic remote model resolution is disabled")
        return str(path.resolve())
    def no_asr(*args, **kwargs):
        raise RuntimeError("Automatic ASR is disabled")
    impl._resolve_model_path = local_only
    impl.OmniVoice.load_asr_model = no_asr
    model = impl.OmniVoice.from_pretrained(args.model, device_map="mps", dtype=torch.float16,
        local_files_only=True, trust_remote_code=False, load_asr=False)
    assert model.sampling_rate == 24000
    torch.manual_seed(20261010)
    np.random.seed(20261010)
    audio = np.asarray(model.generate(text=TEXT, language="te", num_step=32,
                                     normalize_text=False)[0], np.float32)
    if not len(audio) or not np.isfinite(audio).all() or float(np.max(np.abs(audio))) < 1e-6:
        raise RuntimeError("Unusable reference waveform")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, 24000, subtype="FLOAT")
    actual = hashlib.sha256(args.output.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"Regenerated reference differs from selected B; retained at {args.output}. "
                           "The approved reference has not been replaced.")


if __name__ == "__main__":
    main()
