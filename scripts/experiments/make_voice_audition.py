#!/usr/bin/env python3
"""Build a self-contained local listening page from original-speech experiment results.

Only generated WAVs are embedded. No network, downloaded transcript, or production job is read.
The page deliberately hides model identities until the listener reveals them.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path


def build(results: Path, out: Path) -> None:
    report = json.loads(results.read_text())
    storage_id = hashlib.sha256(results.read_bytes()).hexdigest()[:12]
    samples = [s for s in report["samples"] if s.get("path") and Path(s["path"]).is_file()]
    if not samples:
        raise SystemExit("No completed audition WAVs; not creating a misleading empty comparison.")
    models = sorted({s["model"] for s in samples})
    # Keep identities stable when another candidate is added after the first user audition.
    known = {"chatterbox": "A", "omnivoice": "B", "indic-mio": "C", "indic_mio": "C",
             "omnivoice-native": "D", "omnivoice-adapter": "E", "omnivoice-adapter-auto": "F"}
    labels = {m: known[m] for m in models if m in known}
    for model in models:
        if model not in labels:
            labels[model] = next(chr(i) for i in range(65, 91) if chr(i) not in labels.values())
    groups: dict[str, list[dict]] = {}
    for sample in samples:
        groups.setdefault(sample["text"], []).append(sample)
    cards = []
    for number, (text, group) in enumerate(groups.items(), 1):
        voices = []
        for sample in sorted(group, key=lambda s: labels[s["model"]]):
            source = "data:audio/wav;base64," + base64.b64encode(Path(sample["path"]).read_bytes()).decode("ascii")
            label = labels[sample["model"]]
            voices.append(f'''<div class="voice"><span class="label">VOICE {label}</span>
              <audio controls preload="metadata" src="{source}"></audio></div>''')
        options = ''.join(f'<option value="{label}">Voice {label}</option>' for label in sorted({labels[s["model"]] for s in group}))
        cards.append(f'''<article><span class="eyebrow">SAMPLE {number:02d}</span>
          <p lang="te" class="telugu">{html.escape(text)}</p><div class="voices">{''.join(voices)}</div>
          <div class="rating"><label>Sounds more natural <select data-save="choice-{number}">
          <option value="">Choose after listening</option>{options}<option>Neither is good enough</option>
          <option>No clear difference</option></select></label>
          <label>Pronunciation or delivery problems <input data-save="note-{number}" placeholder="Words, endings, pauses…"></label></div></article>''')
    identities = ''.join(f'<li><strong>Voice {labels[m]}</strong>: {html.escape(m)}</li>'
                         for m in sorted(models, key=labels.get))
    conditions = {"chatterbox": "A uses bundled Chatterbox conditioning of unknown native-language provenance.",
                  "omnivoice": "B uses automatic OmniVoice Telugu speech without an English cloning reference.",
                  "omnivoice-native": "D uses a documented human Telugu reference and its supplied transcript.",
                  "omnivoice-adapter": "E is the rejected reference-conditioned trial. Its short greeting has a possible missing final clause; it is not the app default.",
                  "omnivoice-adapter-auto": "F is the actual app configuration: automatic Telugu at 32 steps with seed 20261010 reset per line, followed by timing and watermarking. There is no reference voice, and voice identity can vary."}
    condition_text = " ".join(conditions[m] for m in models if m in conditions)
    acceptance = report.get("listener_note", "No native-speaker acceptance has been recorded.")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Maata · Telugu voice audition</title><style>
:root{color-scheme:light;font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#1d2d2a;background:#f2f5f0}
*{box-sizing:border-box}body{margin:0}main{max-width:1020px;margin:auto;padding:42px 28px 70px}
.brand{font-size:17px;font-weight:750;letter-spacing:.02em;color:#17664e}.eyebrow,.label{font-size:11px;font-weight:750;letter-spacing:.15em;color:#567269}
h1{font-size:42px;letter-spacing:-1.5px;line-height:1.1;margin:32px 0 16px}header p{max-width:760px;color:#52605a;line-height:1.6}
.badge{display:inline-block;font-size:12px;padding:7px 11px;border:1px solid #c5d7cd;border-radius:20px;background:#e6f1e9;margin:5px 6px 16px 0}
article{background:#fff;border:1px solid #dce4dd;border-radius:16px;padding:25px;margin:18px 0;box-shadow:0 8px 24px #1d352706}
.telugu{font-family:"Noto Sans Telugu","Telugu Sangam MN",sans-serif;font-size:22px;line-height:1.9;margin:16px 0 24px}
.voices{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:24px}.voice{display:flex;flex-direction:column;gap:11px}audio{width:100%}
.rating{display:grid;grid-template-columns:1fr 1fr;gap:24px;border-top:1px solid #edf0eb;margin-top:24px;padding-top:20px}
label{display:flex;flex-direction:column;gap:9px;font-size:12px;font-weight:650;color:#51615b}input,select{font:inherit;font-weight:400;border:1px solid #d7dfd8;padding:11px 10px;border-radius:8px;background:#fff;color:#243e32;width:100%}
details{margin-top:28px;font-size:13px;line-height:1.65;color:#52605a}summary{cursor:pointer;font-weight:650;color:#254d3c}
footer{font-size:12px;color:#66736b;margin-top:24px;line-height:1.7}.note{padding:16px 20px;border-left:3px solid #80a78c;background:#e9efe6;border-radius:0 9px 9px 0;font-size:13px;line-height:1.6}
@media(max-width:650px){main{padding:24px 16px}h1{font-size:32px}.rating{grid-template-columns:1fr}.telugu{font-size:20px}}
</style><main><div class="brand">మాట / MAATA</div><header><h1>Which voice sounds like natural Telugu?</h1>
<p>Compare the same original sentences. Listen for clear pronunciation, complete word endings and natural pauses.
Choose with your ears before revealing the model names.</p><span class="badge">Generated locally on this Mac</span>
<span class="badge">Original test text</span><span class="badge">Separate from your queue</span></header>
<div class="note">These are short original test utterances, not a full video dub. Test conditions are listed below;
this comparison does not establish cloning quality or long-video reliability. LISTENER_NOTE</div>
''' + ''.join(cards) + '''<details><summary>Reveal models and test limits</summary><ul>''' + identities + '''</ul>
<p>CONDITION_TEXT Your saved choices stay only in this browser.</p></details>
<footer>Please tell Codex which voice you prefer and any words that sound wrong. Automatic transcription and watermark checks
cannot certify pronunciation or naturalness. These clips have no network dependencies and do not play automatically.</footer></main>
<script>
document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(b!==a)b.pause()})));
document.querySelectorAll('[data-save]').forEach(e=>{const k='maata-voice-audition-2026-10-10-'+e.dataset.save;try{e.value=localStorage.getItem(k)||''}catch{}e.addEventListener('change',()=>{try{localStorage.setItem(k,e.value)}catch{}})});
</script></html>'''
    page = page.replace("maata-voice-audition-2026-10-10-", f"maata-voice-audition-{storage_id}-")
    page = page.replace("LISTENER_NOTE", html.escape(acceptance)).replace("CONDITION_TEXT", html.escape(condition_text))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(json.dumps({"page": str(out.resolve()), "samples": len(samples), "models": labels}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.results, args.out)
