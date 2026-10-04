<script lang="ts">
  import { onMount } from "svelte";
  import TopBar from "./lib/components/TopBar.svelte";
  import Library from "./lib/components/Library.svelte";
  import NewDub from "./lib/components/NewDub.svelte";
  import Job from "./lib/components/Job.svelte";
  import Settings from "./lib/components/Settings.svelte";
  import ClaudeBanner from "./lib/components/ClaudeBanner.svelte";
  import Icon from "./lib/components/Icon.svelte";
  import { app, REMOVE_RETRY_MS } from "./lib/state.svelte";
  import { hear } from "./lib/hear.svelte";
  import { EngineClient, engineUrl, sampleSpeaker } from "./lib/engine";
  import type { ClientMessage } from "./lib/types";

  /**
   * The background dub's app (OFFLINE-RENDER §5): the Library of jobs, New dub for a pasted link, a job's progress with
   * its speaker check, and Settings. The engine runs the jobs and saves the videos; nothing plays here but a speaker's
   * Hear voice sample.
   */
  let client: EngineClient | null = null;
  const send = (m: ClientMessage) => client?.send(m);
  let engineMissing = $state(false);
  let urlInput = $state<HTMLInputElement>();
  let libraryInput = $state<HTMLInputElement>();

  // The theme follows the system (§4: no theme setting).
  $effect(() => {
    const mq = matchMedia("(prefers-color-scheme: dark)");
    const apply = () => (document.documentElement.dataset.theme = mq.matches ? "dark" : "light");
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  });

  // A removal waits for its job's pause to finish: ask again until the engine lets it go.
  $effect(() => {
    if (!Object.keys(app.removing).length) return;
    const id = setInterval(() => app.retryRemoves(send), REMOVE_RETRY_MS);
    return () => clearInterval(id);
  });

  // An error says its piece for a while, then goes.
  $effect(() => {
    if (!app.error) return;
    const id = setTimeout(() => (app.error = null), 9000);
    return () => clearTimeout(id);
  });

  onMount(() => {
    // ?v=<link> (the shell's MAATA_OPEN) opens New dub with that link once the engine says hello.
    app.launch = new URLSearchParams(location.search).get("v");
    const url = engineUrl();
    if (!url) {
      engineMissing = true;
      return;
    }
    client = new EngineClient(url, {
      message: (m) => app.receive(m, send),
      audio: (frame) => {
        const sid = sampleSpeaker(frame.id);
        if (sid) hear.play(sid, frame);
      },
      connection: (c) => (app.connection = c),
    });
    client.connect();
    return () => {
      hear.stop();
      client?.close();
    };
  });

  function onKey(e: KeyboardEvent) {
    const cmd = e.metaKey || e.ctrlKey;
    if (cmd && e.key.toLowerCase() === "l") {
      e.preventDefault();
      const field = app.view === "library" ? libraryInput : urlInput;
      field?.focus();
      field?.select();
    } else if (cmd && e.key === ",") {
      e.preventDefault();
      app.settingsOpen = true;
    }
  }
</script>

<svelte:window onkeydown={onKey} />

<div class="backdrop" aria-hidden="true"><div class="b1"></div><div class="b2"></div><div class="b3"></div><div class="grain"></div></div>

<div class="shell">
  <TopBar onInspect={(u) => app.inspect(u, send)} bind:inputEl={urlInput} />

  <main>
    <ClaudeBanner />
    {#if engineMissing}
      <div class="notice">
        <h2>The Maata engine isn't running</h2>
        <p>Start it with <code>uv run maata-engine --ui app/dist</code> and open the address it prints, or launch the Maata app.</p>
      </div>
    {:else if app.view === "new" && app.video}
      <NewDub {send} />
    {:else if app.view === "job" && app.job}
      <Job {send} />
    {:else}
      <Library {send} bind:inputEl={libraryInput} />
    {/if}
  </main>
</div>

{#if app.error}
  <div class="toast" role="alert">
    <Icon name="alert" size={15} /><span>{app.error.message}</span>
    <button aria-label="Dismiss" onclick={() => (app.error = null)}><Icon name="close" size={14} /></button>
  </div>
{/if}
{#if app.settingsOpen}<Settings {send} />{/if}

<style>
  .shell { position: relative; z-index: 1; height: 100%; display: flex; flex-direction: column; }
  main { flex: 1; overflow: auto; }

  .backdrop { position: fixed; inset: 0; overflow: hidden; z-index: 0; pointer-events: none; }
  .backdrop > div:not(.grain) { position: absolute; border-radius: 50%; filter: blur(90px); opacity: 0.5; }
  .b1 { width: 55vw; height: 55vw; left: -18vw; top: -26vw; background: radial-gradient(circle, rgb(247 183 51 / 0.45), transparent 65%); animation: drift 28s ease-in-out infinite alternate; }
  .b2 { width: 50vw; height: 50vw; right: -20vw; top: -10vw; background: radial-gradient(circle, rgb(216 51 107 / 0.4), transparent 65%); animation: drift 34s ease-in-out infinite alternate-reverse; }
  .b3 { width: 60vw; height: 60vw; left: 20vw; bottom: -40vw; background: radial-gradient(circle, rgb(90 60 200 / 0.35), transparent 65%); animation: drift 40s ease-in-out infinite alternate; }
  :global([data-theme="light"]) .backdrop > div:not(.grain) { opacity: 0.3; }
  .grain { position: absolute; inset: 0; opacity: 0.05; background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>"); }
  @keyframes drift { to { transform: translate(6vw, 4vw) scale(1.08); } }

  .notice { max-width: 560px; margin: 18vh auto; text-align: center; padding: 28px; border-radius: var(--r-xl); background: var(--surface); border: 1px solid var(--border); }
  .notice h2 { margin: 0 0 8px; }
  .notice p { color: var(--text-2); margin: 0; }
  code { font: 12px ui-monospace, monospace; background: var(--surface-strong); padding: 2px 6px; border-radius: 6px; }
  .toast {
    position: fixed; z-index: 30; bottom: 24px; left: 50%; transform: translateX(-50%); max-width: min(640px, calc(100vw - 32px));
    display: flex; align-items: center; gap: 10px; padding: 10px 10px 10px 14px; border-radius: 14px;
    background: var(--bg-raised); border: 1px solid rgb(255 93 93 / 0.45); box-shadow: var(--shadow-lg); color: var(--text);
    animation: up 0.25s var(--ease);
  }
  .toast :global(svg) { color: var(--danger); flex: none; }
  .toast span { user-select: text; -webkit-user-select: text; }
  .toast button { flex: none; border: 0; background: none; color: var(--text-3); cursor: pointer; display: grid; place-items: center; padding: 4px; border-radius: 8px; }
  .toast button:hover { color: var(--text); background: var(--surface-hover); }
  @keyframes up { from { opacity: 0; transform: translate(-50%, 8px); } }
</style>
