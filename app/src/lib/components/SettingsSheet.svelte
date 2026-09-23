<script lang="ts">
  import Icon from "./Icon.svelte";
  import { app } from "../state.svelte";

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
        <div><div class="l">Maximum speed-up</div><div class="h">How much a Telugu line may be sped up to fit its slot.</div></div>
        <div class="range"><input type="range" min="1" max="1.4" step="0.05" bind:value={app.settings.speedCap} /><span class="tabular">{app.settings.speedCap.toFixed(2)}×</span></div>
      </div>
      <label class="field sw"><div><div class="l">Allow gentle slow-downs</div><div class="h">Slow the video slightly when a line runs long.</div></div><input type="checkbox" bind:checked={app.settings.allowSlowdown} /></label>
      <label class="field sw"><div><div class="l">Allow brief pauses</div><div class="h">Hold the frame for a moment instead of cutting speech.</div></div><input type="checkbox" bind:checked={app.settings.allowFreeze} /></label>
      <div class="field">
        <div><div class="l">Longest pause</div></div>
        <div class="range"><input type="range" min="0.5" max="3" step="0.25" bind:value={app.settings.maxPause} /><span class="tabular">{app.settings.maxPause.toFixed(2)} s</span></div>
      </div>
    {:else if tab === "voice"}
      <div class="field">
        <div><div class="l">Translation style</div><div class="h">Everyday spoken Telugu keeps common English terms (Tenglish).</div></div>
        <div class="seg">
          <button class:on={app.settings.style === "colloquial"} onclick={() => (app.settings.style = "colloquial")}>Natural</button>
          <button class:on={app.settings.style === "formal"} onclick={() => (app.settings.style = "formal")}>More formal</button>
        </div>
      </div>
      <div class="field">
        <div><div class="l">Voices</div><div class="h">Clone each speaker's voice, or always use preset Telugu voices.</div></div>
        <div class="seg">
          <button class:on={app.settings.voiceMode === "clone"} onclick={() => (app.settings.voiceMode = "clone")}>Clone speakers</button>
          <button class:on={app.settings.voiceMode === "preset"} onclick={() => (app.settings.voiceMode = "preset")}>Preset voices</button>
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
      </div>
    {:else}
      <label class="field sw"><div><div class="l">Debug HUD</div><div class="h">Lead, drift and per-stage timing over the video.</div></div><input type="checkbox" bind:checked={app.settings.hud} /></label>
      <label class="field sw"><div><div class="l">Check for updates</div><div class="h">The only network use besides YouTube and model downloads.</div></div><input type="checkbox" bind:checked={app.settings.updateCheck} /></label>
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
  .seg { display: flex; padding: 3px; border-radius: 10px; background: var(--surface); border: 1px solid var(--border); flex: none; }
  .seg button { border: 0; background: transparent; padding: 6px 12px; border-radius: 8px; cursor: pointer; font-weight: 560; color: var(--text-2); white-space: nowrap; }
  .seg button.on { background: var(--surface-strong); color: var(--text); }
  .models { padding: 12px 0; }
  @keyframes fade { from { opacity: 0; } }
  @keyframes pop { from { opacity: 0; transform: translate(-50%, -48%) scale(0.98); } }
</style>
