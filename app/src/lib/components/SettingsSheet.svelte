<script lang="ts">
  import Icon from "./Icon.svelte";
  import { app } from "../state.svelte";
  import {
    LOOKAHEAD_OPTIONS, PREPARE_OPTIONS, SPEED_CAP_MAX, SPEED_CAP_MIN, SPEED_CAP_STEP, STYLE_OPTIONS, TTS_SCRIPT_OPTIONS,
  } from "../settings";
  import { CLONE_STRENGTH_OPTIONS } from "../voice";

  let tab = $state<"timing" | "voice" | "models" | "advanced">("timing");
  const tabs = [
    { id: "timing", label: "Timing" },
    { id: "voice", label: "Voice & language" },
    { id: "models", label: "Models & storage" },
    { id: "advanced", label: "Advanced" },
  ] as const;

  function close() {
    app.saveSettings();
    app.settingsOpen = false;
  }
</script>

<svelte:window onkeydown={(e) => e.key === "Escape" && close()} />

<div class="scrim" role="presentation" onclick={close}></div>
<div class="sheet" role="dialog" aria-modal="true" aria-labelledby="settings-title">
  <header>
    <h2 id="settings-title">Settings</h2>
    <button class="x" aria-label="Close settings" onclick={close}><Icon name="close" /></button>
  </header>
  <nav>
    {#each tabs as t (t.id)}
      <button class:on={tab === t.id} onclick={() => (tab = t.id)}>{t.label}</button>
    {/each}
  </nav>
  <div class="body">
    {#if tab === "timing"}
      <div class="field">
        <div><div class="l">Prepare before playing</div><div class="h">Telugu to have ready before the video starts, so it plays without stopping.</div></div>
        <div class="seg" role="radiogroup" aria-label="Prepare before playing">
          {#each PREPARE_OPTIONS as v (v)}
            <button role="radio" aria-checked={app.settings.prepareAhead === v} class:on={app.settings.prepareAhead === v} onclick={() => (app.settings.prepareAhead = v)}>{v / 60} min</button>
          {/each}
        </div>
      </div>
      <div class="field">
        <div><div class="l">Work ahead</div><div class="h">How far past the playhead the engine keeps dubbing. Applies to the next video.</div></div>
        <div class="seg" role="radiogroup" aria-label="Work ahead">
          {#each LOOKAHEAD_OPTIONS as v (v)}
            <button role="radio" aria-checked={app.settings.lookahead === v} class:on={app.settings.lookahead === v} onclick={() => (app.settings.lookahead = v)}>{v / 60} min</button>
          {/each}
        </div>
      </div>
      <div class="field">
        <div><div class="l">Maximum speed-up</div><div class="h">How much a Telugu line may be sped up to keep pace with the video. Above 1.2× speech starts to sound rushed.</div></div>
        <div class="range">
          <input type="range" min={SPEED_CAP_MIN} max={SPEED_CAP_MAX} step={SPEED_CAP_STEP} aria-label="Maximum speed-up" bind:value={app.settings.speedCap} />
          <span class="tabular">{app.settings.speedCap.toFixed(2)}×</span>
        </div>
      </div>
      <label class="field sw off">
        <div>
          <div class="l">Allow gentle slow-downs <span class="tag">Off</span></div>
          <div class="h">The video now always plays at its own speed. Each Telugu line is fitted to the original timing instead, so the dub stays in step with the picture.</div>
        </div>
        <input type="checkbox" checked={false} disabled />
      </label>
      <label class="field sw">
        <div><div class="l">Allow brief pauses</div><div class="h">Rarely, hold the picture for a split second when a Telugu line can't fit, instead of letting the dub fall behind.</div></div>
        <input type="checkbox" bind:checked={app.settings.allowFreeze} />
      </label>
      <p class="foot">Speed-up and pause settings apply to the next video you open.</p>
    {:else if tab === "voice"}
      <div class="field stack">
        <div><div class="l">Translation style</div><div class="h">Applies to the next video you open.</div></div>
        <div class="choices" role="radiogroup" aria-label="Translation style">
          {#each STYLE_OPTIONS as st (st.id)}
            <button role="radio" aria-checked={app.settings.style === st.id} class="choice" class:on={app.settings.style === st.id} onclick={() => (app.settings.style = st.id)}>
              <span class="dot" aria-hidden="true"></span>
              <span><span class="cl">{st.label}</span><span class="ch">{st.hint}</span></span>
            </button>
          {/each}
        </div>
      </div>
      <div class="field stack">
        <div>
          <div class="l">Voice match</div>
          <div class="h">
            How closely the cloned voices follow the original speakers, against how natural the Telugu sounds. One setting for
            every speaker; it applies to the next video you open. The Speakers panel shows the match each voice is actually built with.
          </div>
        </div>
        <div class="choices three" role="radiogroup" aria-label="Voice match">
          {#each CLONE_STRENGTH_OPTIONS as o (o.id)}
            <button role="radio" aria-checked={app.settings.cloneStrength === o.id} class="choice" class:on={app.settings.cloneStrength === o.id} onclick={() => (app.settings.cloneStrength = o.id)}>
              <span class="dot" aria-hidden="true"></span>
              <span><span class="cl">{o.label}</span><span class="ch">{o.hint}</span></span>
            </button>
          {/each}
        </div>
      </div>
      <div class="field stack">
        <div>
          <div class="l">English words in the Telugu voice</div>
          <div class="h">How the voice is given the English words a line keeps. Applies to the next video you open.</div>
        </div>
        <div class="choices" role="radiogroup" aria-label="English words in the Telugu voice">
          {#each TTS_SCRIPT_OPTIONS as o (o.id)}
            <button role="radio" aria-checked={app.settings.ttsScript === o.id} class="choice" class:on={app.settings.ttsScript === o.id} onclick={() => (app.settings.ttsScript = o.id)}>
              <span class="dot" aria-hidden="true"></span>
              <span><span class="cl">{o.label}</span><span class="ch">{o.hint}</span></span>
            </button>
          {/each}
        </div>
      </div>
      <div class="field">
        <div><div class="l">Appearance</div></div>
        <div class="seg">
          {#each ["dark", "light", "system"] as const as th (th)}
            <button class:on={app.settings.theme === th} onclick={() => (app.settings.theme = th)}>{th[0]!.toUpperCase() + th.slice(1)}</button>
          {/each}
        </div>
      </div>
    {:else if tab === "models"}
      <div class="models">
        <div class="m"><div class="l">Engine</div><div class="h">{app.demo ? "Demo engine — no models loaded" : `${app.backend} · ${app.device}`}</div></div>
        <p class="h">Models download once, are verified by checksum, and never leave your Mac. Model management lands in Phase 2.</p>
        {#if app.claude}
          <div class="m">
            <div class="l">Translation</div>
            <div class="h">
              {app.claude.installed ? `Claude Code ${app.claude.version ?? ""}` : "Claude Code not found"}
              {app.claude.signedIn === true ? "· signed in" : app.claude.signedIn === false ? "· not signed in" : ""}
              · {app.claude.model}. Transcript text only; audio stays on this Mac.
            </div>
          </div>
        {/if}
      </div>
    {:else}
      <label class="field sw"><div><div class="l">Debug HUD</div><div class="h">Lead, drift and per-stage timing over the video.</div></div><input type="checkbox" bind:checked={app.settings.hud} /></label>
      <label class="field sw"><div><div class="l">Check for updates</div><div class="h">The only network use besides YouTube, model downloads and your Claude Code translating the transcript.</div></div><input type="checkbox" bind:checked={app.settings.updateCheck} /></label>
    {/if}
  </div>
</div>

<style>
  .scrim { position: fixed; inset: 0; background: rgb(5 4 10 / 0.55); backdrop-filter: blur(6px); z-index: 20; animation: fade 0.2s ease; }
  .sheet {
    position: fixed; top: 50%; left: 50%; width: min(620px, 92vw); max-height: 80vh; transform: translate(-50%, -50%);
    display: flex; flex-direction: column; z-index: 21; border-radius: var(--r-xl);
    background: var(--bg-raised); border: 1px solid var(--border-strong); box-shadow: var(--shadow-lg);
    animation: pop 0.25s var(--ease);
  }
  header { display: flex; justify-content: space-between; align-items: center; padding: 18px 20px 6px; }
  h2 { margin: 0; font-size: 18px; letter-spacing: -0.01em; }
  .x { width: 32px; height: 32px; border-radius: 9px; border: 0; background: transparent; color: var(--text-2); cursor: pointer; display: grid; place-items: center; }
  .x:hover { background: var(--surface-hover); }
  nav { display: flex; gap: 4px; padding: 8px 16px; border-bottom: 1px solid var(--border); }
  nav button { border: 0; background: transparent; padding: 7px 12px; border-radius: 9px; color: var(--text-2); cursor: pointer; font-weight: 560; }
  nav button.on { background: var(--surface-strong); color: var(--text); }
  .body { padding: 8px 20px 20px; overflow: auto; }
  .field { display: flex; justify-content: space-between; align-items: center; gap: 20px; padding: 14px 0; border-bottom: 1px solid var(--border); }
  .field:last-child { border-bottom: 0; }
  .l { font-weight: 600; }
  .h { color: var(--text-3); font-size: 12.5px; margin-top: 2px; }
  .range { display: flex; align-items: center; gap: 10px; }
  .range span { width: 52px; text-align: right; font-weight: 600; }
  input[type="range"] { accent-color: var(--vermilion); width: 150px; }
  .sw input { width: 18px; height: 18px; accent-color: var(--vermilion); }
  .sw.off { cursor: default; }
  .sw.off .l { color: var(--text-2); }
  .sw input:disabled { opacity: 0.45; }
  .tag {
    display: inline-block; margin-left: 6px; padding: 0 6px; border-radius: 99px; vertical-align: 1px;
    font-size: 10.5px; font-weight: 650; letter-spacing: 0.02em; color: var(--text-3); border: 1px solid var(--border-strong);
  }
  .foot { margin: 12px 0 0; color: var(--text-3); font-size: 12px; }
  .seg { display: flex; padding: 3px; border-radius: 10px; background: var(--surface); border: 1px solid var(--border); flex: none; }
  .seg button { border: 0; background: transparent; padding: 6px 12px; border-radius: 8px; cursor: pointer; font-weight: 560; color: var(--text-2); white-space: nowrap; }
  .seg button.on { background: var(--surface-strong); color: var(--text); }
  .models { padding: 12px 0; }
  .field.stack { flex-direction: column; align-items: stretch; gap: 10px; }
  .choices { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .choices.three { grid-template-columns: repeat(3, 1fr); }
  .choice {
    display: flex; gap: 10px; align-items: flex-start; text-align: left; padding: 11px 12px; border-radius: var(--r-md); cursor: pointer;
    background: var(--surface); border: 1px solid var(--border); transition: border-color 0.2s var(--ease), background 0.2s var(--ease);
  }
  .choice:hover { background: var(--surface-hover); }
  .choice.on { border-color: rgb(242 96 60 / 0.55); background: rgb(242 96 60 / 0.08); }
  .choice .dot { flex: none; width: 14px; height: 14px; margin-top: 2px; border-radius: 50%; border: 1.5px solid var(--border-strong); }
  .choice.on .dot { border: 4px solid var(--vermilion); }
  .cl { display: block; font-weight: 600; }
  .ch { display: block; color: var(--text-3); font-size: 12px; margin-top: 2px; }
  @media (max-width: 560px) { .choices, .choices.three { grid-template-columns: 1fr; } }
  @keyframes fade { from { opacity: 0; } }
  @keyframes pop { from { opacity: 0; transform: translate(-50%, -48%) scale(0.98); } }
</style>
