<script lang="ts">
  import Icon from "./Icon.svelte";
  import { app } from "../state.svelte";

  let { onInspect, inputEl = $bindable() }: { onInspect: (url: string) => void; inputEl?: HTMLInputElement } = $props();
  let value = $state("");
  let dragging = $state(false);

  function submit(e?: Event) {
    e?.preventDefault();
    if (value.trim()) {
      onInspect(value.trim());
      value = "";
    }
  }
  async function paste() {
    try {
      value = (await navigator.clipboard.readText()).trim();
      submit();
    } catch {
      inputEl?.focus();
    }
  }
  function onDrop(e: DragEvent) {
    e.preventDefault();
    dragging = false;
    const text = e.dataTransfer?.getData("text/uri-list") || e.dataTransfer?.getData("text/plain") || "";
    if (text) {
      value = text.trim().split("\n")[0] ?? "";
      submit();
    }
  }
  const backendLabel = $derived(
    app.demo ? "Demo engine" : app.backend === "apple" ? `Apple Silicon · ${app.ttsModel}` : app.backend === "cuda" ? "NVIDIA · CUDA" : "Connecting…",
  );
  // The Library has its own big paste field.
  const showField = $derived(app.view !== "library");
</script>

<header class="bar" data-tauri-drag-region>
  <button class="brand" onclick={() => app.home()} aria-label="Maata: your dubs">
    <span class="mark te" aria-hidden="true">మా</span>
    <span class="word">Maata</span>
  </button>

  {#if showField}
    <form class="url" class:dragging novalidate onsubmit={submit} ondragover={(e) => { e.preventDefault(); dragging = true; }} ondragleave={() => (dragging = false)} ondrop={onDrop}>
      <span class="lead"><Icon name="link" size={16} /></span>
      <input
        bind:this={inputEl}
        bind:value
        type="url"
        spellcheck="false"
        autocomplete="off"
        placeholder="Paste a YouTube link to dub it into Telugu"
        aria-label="YouTube link"
      />
      <kbd class="hint" aria-hidden="true">⌘L</kbd>
      {#if value}
        <button type="submit" class="go" aria-label="Set up a dub of this video"><Icon name="arrow" size={16} /></button>
      {:else}
        <button type="button" class="go ghost" onclick={paste} aria-label="Paste link"><Icon name="paste" size={16} /></button>
      {/if}
    </form>
  {:else}
    <div data-tauri-drag-region></div>
  {/if}

  <div class="right" data-tauri-drag-region>
    <div class="badge" title={app.voiceMode === "native" ? "AI-generated natural Telugu speech. Source voices are not cloned; the automatic voice can vary between lines." : "The Telugu voices are AI-generated from the original speakers"}>
      <Icon name="sparkle" size={13} />
      <span>AI dub</span>
      <span class="te sep">తెలుగు</span>
    </div>
    <div class="engine" class:live={app.connection === "open"} title={app.device ? `${app.ttsModel} · Engine on ${app.device}` : ""}>
      <span class="dot"></span>
      <Icon name="cpu" size={14} />
      <span>{app.connection === "closed" ? "Reconnecting…" : backendLabel}</span>
    </div>
    <button class="icon-btn" aria-label="Settings" title="Settings (⌘,)" onclick={() => (app.settingsOpen = true)}><Icon name="settings" /></button>
  </div>
</header>

<style>
  .bar {
    height: 64px;
    display: grid;
    grid-template-columns: 1fr minmax(360px, 640px) 1fr;
    align-items: center;
    gap: 16px;
    padding: 0 20px 0 88px; /* room for macOS traffic lights */
    position: relative;
    z-index: 5;
  }
  .brand { display: flex; align-items: center; gap: 10px; justify-self: start; padding: 4px 6px; margin-left: -6px; border: 0; background: none; cursor: pointer; border-radius: 12px; }
  .brand:hover .word { color: var(--text); }
  .mark {
    width: 30px; height: 30px; border-radius: 10px;
    display: grid; place-items: center;
    background: var(--accent-grad);
    color: #1a0f08; font-weight: 700; font-size: 14px;
    box-shadow: var(--accent-glow);
  }
  .word { font-weight: 650; letter-spacing: -0.01em; font-size: 16px; color: var(--text); }

  .url {
    display: flex; align-items: center; gap: 8px;
    height: 42px; padding: 0 6px 0 14px;
    border-radius: 999px;
    background: var(--surface);
    border: 1px solid var(--border);
    backdrop-filter: var(--blur);
    transition: border-color 0.2s var(--ease), background 0.2s var(--ease), box-shadow 0.2s var(--ease);
  }
  .url:focus-within, .url.dragging { border-color: rgb(242 96 60 / 0.55); background: var(--surface-hover); box-shadow: var(--accent-glow); }
  .lead { color: var(--text-3); display: grid; }
  input { flex: 1; min-width: 0; border: 0; outline: 0; background: transparent; font-size: 14px; user-select: text; -webkit-user-select: text; }
  input::placeholder { color: var(--text-3); }
  .hint { font: 500 11px var(--font-sans); color: var(--text-3); border: 1px solid var(--border); border-radius: 6px; padding: 2px 6px; }
  .go {
    width: 32px; height: 32px; border-radius: 999px; border: 0; cursor: pointer;
    display: grid; place-items: center;
    background: var(--accent-grad); color: #1a0f08;
    transition: transform 0.15s var(--ease);
  }
  .go:hover { transform: scale(1.06); }
  .go.ghost { background: var(--surface-strong); color: var(--text-2); }

  .right { display: flex; justify-content: flex-end; align-items: center; gap: 10px; }
  .badge, .engine {
    display: flex; align-items: center; gap: 6px;
    height: 30px; padding: 0 12px; border-radius: 999px;
    background: var(--surface); border: 1px solid var(--border);
    color: var(--text-2); font-size: 12px; font-weight: 600; white-space: nowrap;
  }
  .badge { color: var(--text); }
  .badge :global(svg) { color: var(--turmeric); }
  .badge .sep { font-weight: 500; color: var(--text-2); padding-left: 6px; border-left: 1px solid var(--border-strong); }
  .engine { font-weight: 500; }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--text-3); }
  .engine.live .dot { background: var(--ok); box-shadow: 0 0 10px var(--ok); }
  .icon-btn {
    width: 34px; height: 34px; border-radius: 10px; border: 1px solid transparent;
    background: transparent; color: var(--text-2); cursor: pointer; display: grid; place-items: center;
  }
  .icon-btn:hover { background: var(--surface-hover); color: var(--text); }
  @media (max-width: 1100px) { .engine span:last-child { display: none; } }
</style>
