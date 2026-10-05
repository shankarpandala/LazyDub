<script lang="ts">
  import Icon from "./Icon.svelte";
  import { app } from "../state.svelte";
  import type { ClientMessage } from "../types";

  /**
   * Settings (OFFLINE-RENDER §4, §5): the one the engine keeps, where the dubbed videos are saved. No theme (Maata
   * follows the system) and no stored defaults (New dub starts from the last job's options).
   */
  let { send }: { send: (m: ClientMessage) => void } = $props();
  let folder = $state(app.settings?.outputDir ?? "");
  /** The settings a save was sent with: the engine's next `settings` (a new object) is its answer. */
  let pending = $state.raw<object | null | undefined>(undefined);
  let saved = $state(false);
  // The engine's answer confirms the save and shows the folder as it keeps it (~ expanded, no trailing slash);
  // a refused folder gets an error instead, and the field keeps what was typed.
  $effect(() => {
    const now = app.settings;
    if (pending !== undefined && now && now !== pending) {
      folder = now.outputDir;
      saved = true;
      pending = undefined;
    }
  });
  $effect(() => {
    if (app.error && pending !== undefined) pending = undefined;
  });
  const close = () => (app.settingsOpen = false);
  const focusOnMount = (node: HTMLElement) => node.focus();

  function save(e: Event) {
    e.preventDefault();
    if (!folder.trim()) return;
    saved = false;
    pending = app.settings;
    app.saveSettings(folder, send);
  }
</script>

<svelte:window onkeydown={(e) => e.key === "Escape" && close()} />

<div class="scrim" role="presentation" onclick={close}></div>
<div class="sheet" role="dialog" aria-modal="true" aria-labelledby="settings-title">
  <header>
    <h2 id="settings-title">Settings</h2>
    <button class="icon-btn" aria-label="Close" onclick={close}><Icon name="close" /></button>
  </header>

  <form class="field" onsubmit={save}>
    <label for="out-dir">Save dubbed videos in</label>
    <div class="row">
      <span class="lead"><Icon name="folder" size={16} /></span>
      <input id="out-dir" bind:value={folder} spellcheck="false" autocomplete="off" placeholder="/Users/you/Movies/Maata" use:focusOnMount />
      <button class="btn primary" type="submit" disabled={!folder.trim() || folder.trim() === app.settings?.outputDir}>Save</button>
    </div>
    <p class="hint">
      {#if saved && folder.trim() === app.settings?.outputDir}<span class="ok"><Icon name="check" size={12} /> Saved. New dubs go there; a dub already running keeps its folder.</span>
      {:else}A full path. Maata makes the folder if it isn't there.{/if}
    </p>
  </form>

  <section class="about">
    <h3>Notifications</h3>
    <p>macOS shows Maata's notifications under <strong>Script Editor</strong>: allow Script Editor to notify in System Settings ▸ Notifications to see when speakers are found, a dub is done, or one stops.</p>
    <h3>Appearance</h3>
    <p>Maata follows your Mac's light or dark appearance.</p>
    <h3>Engine</h3>
    <p>{app.demo ? "The demo engine: no models, a synthetic video." : `${app.backend || "…"}${app.device ? ` on ${app.device}` : ""}`}</p>
    {#if !app.demo && app.claude}
      <h3>Translation</h3>
      <p>Codex · {app.claude.model}</p>
    {/if}
  </section>
</div>

<style>
  .scrim { position: fixed; inset: 0; z-index: 20; background: rgb(5 4 10 / 0.5); backdrop-filter: blur(4px); animation: fade 0.2s var(--ease); }
  .sheet {
    position: fixed; z-index: 21; top: 76px; left: 50%; transform: translateX(-50%); width: min(560px, calc(100vw - 32px));
    max-height: calc(100vh - 100px); overflow: auto; padding: 20px 22px 22px; border-radius: var(--r-xl);
    background: var(--bg-raised); border: 1px solid var(--border-strong); box-shadow: var(--shadow-lg); animation: rise 0.25s var(--ease);
  }
  header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
  h2 { margin: 0; font-size: 18px; }
  h3 { margin: 16px 0 4px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-2); }
  .about p { margin: 0; color: var(--text-2); font-size: 13px; }
  .field label { display: block; font-weight: 600; margin-bottom: 8px; }
  .row { display: flex; align-items: center; gap: 8px; padding: 0 5px 0 12px; height: 44px; border-radius: 14px; background: var(--surface); border: 1px solid var(--border-strong); }
  .row:focus-within { border-color: rgb(242 96 60 / 0.55); box-shadow: var(--accent-glow); }
  .lead { color: var(--text-3); display: grid; }
  input { flex: 1; min-width: 0; border: 0; outline: 0; background: transparent; font: 13px ui-monospace, monospace; user-select: text; -webkit-user-select: text; }
  .hint { margin: 8px 0 0; font-size: 12px; color: var(--text-3); }
  .ok { display: inline-flex; align-items: center; gap: 5px; color: var(--ok); }
  .btn { height: 34px; padding: 0 16px; border-radius: 10px; border: 0; cursor: pointer; font-weight: 650; }
  .btn.primary { background: var(--accent-grad); color: #1a0f08; }
  .btn:disabled { opacity: 0.45; cursor: default; }
  .icon-btn { width: 34px; height: 34px; border-radius: 10px; border: 0; background: transparent; color: var(--text-2); cursor: pointer; display: grid; place-items: center; }
  .icon-btn:hover { background: var(--surface-hover); color: var(--text); }
  @keyframes rise { from { opacity: 0; transform: translate(-50%, -6px); } }
  @keyframes fade { from { opacity: 0; } }
</style>
