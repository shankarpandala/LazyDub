<script lang="ts">
  import Icon from "./Icon.svelte";
  import StatusChip from "./StatusChip.svelte";
  import Thumb from "./Thumb.svelte";
  import { app } from "../state.svelte";
  import { engineAsset } from "../engine";
  import { chip, isActive, live } from "../jobs";
  import type { ClientMessage, JobItem } from "../types";

  let { send, inputEl = $bindable() }: { send: (m: ClientMessage) => void; inputEl?: HTMLInputElement } = $props();
  let value = $state("");
  let dragging = $state(false);
  /** The job whose removal is being confirmed in its card. */
  let confirming = $state<string | null>(null);
  // The chips' "resumed 06:40" and the countdowns move.
  let now = $state(Date.now());
  $effect(() => {
    const id = setInterval(() => (now = Date.now()), 15_000);
    return () => clearInterval(id);
  });

  const steps = [
    { icon: "wave", title: "Listens", body: "Finds who speaks when, and what they say, over the whole video." },
    { icon: "sparkle", title: "Translates", body: "Into the Telugu people speak every day, with your own Claude Code." },
    { icon: "user", title: "Speaks", body: "In a clone of each speaker's voice, over the original music and sounds." },
    { icon: "film", title: "Saves", body: "A video file with Telugu audio and Telugu and English subtitles." },
  ] as const;

  function submit(e?: Event) {
    e?.preventDefault();
    if (value.trim()) {
      app.inspect(value, send);
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

  const statusOf = (j: JobItem) => app.renders[j.videoId]?.status ?? j.status;
  const resetsAt = (j: JobItem) => (app.claudeProblem?.videoId === j.videoId ? app.claudeProblem.resetsAt ?? null : null);
  const focusOnMount = (node: HTMLElement) => node.focus();
</script>

<section class="library">
  <div class="headline">
    <h1><span class="te grad-text word">మాట</span></h1>
    <p class="tag">Any YouTube video, dubbed into Telugu in the speakers' own voices.</p>
    <p class="sub">Maata dubs the whole video on your Mac and saves it as a video file. Only the transcript, as text, goes to your own Claude Code.</p>
  </div>

  <form class="paste" class:dragging novalidate onsubmit={submit} ondragover={(e) => { e.preventDefault(); dragging = true; }} ondragleave={() => (dragging = false)} ondrop={onDrop}>
    <span class="lead"><Icon name="link" size={18} /></span>
    <input
      bind:this={inputEl}
      bind:value
      type="url"
      spellcheck="false"
      autocomplete="off"
      placeholder="Paste a YouTube link"
      aria-label="YouTube link"
    />
    {#if app.inspecting}
      <span class="spin" aria-label="Looking up the video"></span>
    {:else if value}
      <button type="submit" class="go">Next <Icon name="arrow" size={16} /></button>
    {:else}
      <button type="button" class="go ghost" onclick={paste}><Icon name="paste" size={16} /> Paste</button>
    {/if}
  </form>

  {#if app.jobs.length}
    <div class="label">Your dubs</div>
    <ul class="jobs">
      {#each app.jobs as j (j.videoId)}
        {@const lj = live(j, app.renders[j.videoId])}
        {@const st = statusOf(j)}
        {@const out = lj.output}
        {@const removing = j.videoId in app.removing}
        <li class="job" class:removing>
          <button class="open thumb-btn" onclick={() => app.openJob(j.videoId)} aria-label={`Open ${j.title || j.videoId}`}>
            <Thumb src={j.thumb ? engineAsset(j.thumb) : null} duration={j.duration} title={j.title} />
          </button>
          <div class="info">
            <button class="open title" onclick={() => app.openJob(j.videoId)}>{j.title || j.videoId}</button>
            <div class="meta">{j.channel}</div>
            <StatusChip chip={chip(lj, now, resetsAt(j))} />
            {#if st === "running" && lj.eta != null}
              {@const r = app.renders[j.videoId]}
              {#if r}
                {@const done = r.stages.filter((s) => s.state === "done").length}
                <div class="overall" aria-hidden="true"><div style={`width:${(100 * done) / Math.max(1, r.stages.length)}%`}></div></div>
              {/if}
            {/if}
          </div>
          <div class="actions">
            {#if st === "done" && out && !out.missing}
              <button class="btn primary" onclick={() => app.openOutput(j.videoId, false, send)}><Icon name="play" size={13} /> Open video</button>
              <button class="btn" onclick={() => app.openOutput(j.videoId, true, send)}><Icon name="folder" size={14} /> Show in Finder</button>
            {:else if st === "done" && out?.missing}
              <button class="btn" onclick={() => app.openOutput(j.videoId, true, send)}><Icon name="folder" size={14} /> Show folder</button>
              <button class="btn" onclick={() => app.resume(j.videoId, send)}><Icon name="redo" size={14} /> Save again</button>
            {/if}
            {#if removing}
              <!-- (paused first, then removed) -->
            {:else if isActive(st) || lj.position != null}
              <button class="btn" onclick={() => app.pause(j.videoId, send)}><Icon name="pause" size={12} /> Pause</button>
            {:else if j.expired}
              <button class="btn" onclick={() => app.continueJob(j, send)}><Icon name="redo" size={14} /> Dub again</button>
            {:else if st === "paused" || st === "interrupted" || st === "failed"}
              <button class="btn" onclick={() => app.continueJob(j, send)}><Icon name="play" size={12} /> Resume</button>
            {/if}
            {#if removing}
              <span class="muted">Removing…</span>
            {:else}
              <button class="icon-btn" aria-label={`Remove ${j.title || j.videoId}`} title="Remove" onclick={() => (confirming = j.videoId)}><Icon name="trash" size={16} /></button>
            {/if}
          </div>
          {#if confirming === j.videoId}
            <div class="confirm" role="alertdialog" tabindex="-1" aria-labelledby={`rm-${j.videoId}`}
                 onkeydown={(e) => e.key === "Escape" && (confirming = null)}>
              <p class="c-title" id={`rm-${j.videoId}`}>Remove this dub from Maata?</p>
              <p class="c-body">
                {isActive(st) ? "It is paused first. " : ""}Its voices and work files go; its translations stay, so dubbing it again
                needs no new Claude calls. {out ? "The saved video stays in its folder." : ""}
              </p>
              <div class="c-actions">
                <button class="btn" use:focusOnMount onclick={() => (confirming = null)}>Cancel</button>
                <button class="btn danger" onclick={() => { app.remove(j, send); confirming = null; }}>Remove</button>
              </div>
            </div>
          {/if}
        </li>
      {/each}
    </ul>
  {:else}
    <div class="steps">
      {#each steps as s, i (s.title)}
        <div class="step" style={`--d:${i * 90}ms`}>
          <div class="ic"><Icon name={s.icon} size={18} /></div>
          <div>
            <div class="t">{s.title}</div>
            <div class="b">{s.body}</div>
          </div>
        </div>
      {/each}
    </div>
  {/if}

  <footer class="notes">
    <p><Icon name="clock" size={14} /> Maata keeps dubbing while its window is closed. Quitting Maata pauses; it continues when you open it again.</p>
    <p><Icon name="alert" size={14} /> Keep the lid open and the Mac plugged in; the screen can turn off.</p>
  </footer>
</section>

<style>
  .library { max-width: 1040px; margin: 0 auto; padding: 5vh 32px 40px; display: flex; flex-direction: column; gap: 28px; }
  .headline { text-align: center; }
  h1 { margin: 0; line-height: 1; }
  .word { font-size: clamp(80px, 12vw, 148px); font-weight: 700; letter-spacing: -0.02em; filter: drop-shadow(0 10px 40px rgb(242 96 60 / 0.35)); }
  .tag { margin: 16px 0 6px; font-size: 22px; font-weight: 560; letter-spacing: -0.015em; }
  .sub { margin: 0 auto; max-width: 640px; color: var(--text-2); font-size: 15px; }

  .paste {
    display: flex; align-items: center; gap: 10px; width: min(680px, 100%); margin: 0 auto;
    height: 56px; padding: 0 8px 0 20px; border-radius: 999px;
    background: var(--surface); border: 1px solid var(--border-strong); backdrop-filter: var(--blur);
    transition: border-color 0.2s var(--ease), box-shadow 0.2s var(--ease), background 0.2s var(--ease);
  }
  .paste:focus-within, .paste.dragging { border-color: rgb(242 96 60 / 0.6); background: var(--surface-hover); box-shadow: var(--accent-glow); }
  .lead { color: var(--text-3); display: grid; }
  .paste input { flex: 1; min-width: 0; border: 0; outline: 0; background: transparent; font-size: 16px; user-select: text; -webkit-user-select: text; }
  .paste input::placeholder { color: var(--text-3); }
  .go {
    display: inline-flex; align-items: center; gap: 6px; height: 40px; padding: 0 18px; border-radius: 999px; border: 0; cursor: pointer;
    background: var(--accent-grad); color: #1a0f08; font-weight: 650; transition: transform 0.15s var(--ease);
  }
  .go:hover { transform: scale(1.03); }
  .go.ghost { background: var(--surface-strong); color: var(--text-2); font-weight: 560; }
  .spin { width: 22px; height: 22px; margin-right: 10px; border-radius: 50%; border: 2px solid var(--border-strong); border-top-color: var(--vermilion); animation: spin 0.8s linear infinite; }

  .label { color: var(--text-2); font-size: 12px; font-weight: 650; text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: -16px; }
  .jobs { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 10px; }
  .job {
    display: grid; grid-template-columns: 176px minmax(0, 1fr) auto; gap: 16px; align-items: center; padding: 12px;
    border-radius: var(--r-lg); background: var(--surface); border: 1px solid var(--border); backdrop-filter: var(--blur);
    animation: rise 0.4s var(--ease) both; transition: border-color 0.2s var(--ease), opacity 0.2s;
  }
  .job:hover { border-color: var(--border-strong); }
  .job.removing { opacity: 0.55; }
  .open { padding: 0; border: 0; background: none; cursor: pointer; text-align: left; }
  .thumb-btn { display: block; border-radius: var(--r-md); transition: transform 0.2s var(--ease); }
  .thumb-btn:hover { transform: scale(1.02); }
  .info { min-width: 0; display: flex; flex-direction: column; align-items: flex-start; gap: 6px; }
  .title { font-weight: 620; font-size: 15px; line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2; line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
  .title:hover { text-decoration: underline; text-decoration-color: var(--border-strong); text-underline-offset: 3px; }
  .meta { color: var(--text-3); font-size: 12.5px; margin-top: -4px; }
  .overall { width: min(320px, 100%); height: 3px; border-radius: 99px; background: var(--surface-strong); overflow: hidden; }
  .overall div { height: 100%; background: var(--accent-grad); border-radius: 99px; transition: width 0.6s var(--ease); }
  .actions { display: flex; align-items: center; justify-content: flex-end; gap: 8px; flex-wrap: wrap; max-width: 340px; }
  .muted { color: var(--text-3); font-size: 12px; }
  .btn {
    display: inline-flex; align-items: center; gap: 6px; height: 32px; padding: 0 13px; border-radius: 999px; cursor: pointer;
    font-size: 12.5px; font-weight: 600; color: var(--text); background: var(--surface-strong); border: 1px solid var(--border-strong);
    white-space: nowrap;
  }
  .btn:hover { background: var(--surface-hover); }
  .btn.primary { background: var(--accent-grad); color: #1a0f08; border-color: transparent; }
  .btn.primary:hover { filter: brightness(1.05); }
  .btn.danger { background: var(--danger); color: #1f0505; border-color: transparent; }
  .icon-btn { width: 32px; height: 32px; border-radius: 10px; border: 1px solid transparent; background: transparent; color: var(--text-3); cursor: pointer; display: grid; place-items: center; }
  .icon-btn:hover { background: var(--surface-hover); color: var(--danger); }
  .confirm {
    grid-column: 1 / -1; padding: 12px 14px; border-radius: var(--r-md); background: var(--bg-raised);
    border: 1px solid rgb(255 93 93 / 0.4); animation: rise 0.2s var(--ease);
  }
  .c-title { margin: 0; font-weight: 620; }
  .c-body { margin: 4px 0 10px; font-size: 12.5px; color: var(--text-2); }
  .c-actions { display: flex; justify-content: flex-end; gap: 8px; }

  .steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
  .step {
    display: flex; gap: 14px; align-items: flex-start; padding: 18px;
    border-radius: var(--r-lg); background: var(--surface); border: 1px solid var(--border); backdrop-filter: var(--blur);
    animation: rise 0.7s var(--ease) both; animation-delay: var(--d);
  }
  .ic { width: 36px; height: 36px; border-radius: 11px; display: grid; place-items: center; color: var(--turmeric); background: rgb(247 183 51 / 0.1); border: 1px solid rgb(247 183 51 / 0.2); flex: none; }
  .t { font-weight: 600; margin-bottom: 2px; }
  .b { color: var(--text-2); font-size: 13px; }

  .notes { display: flex; flex-direction: column; align-items: center; gap: 6px; margin-top: 4px; }
  .notes p { display: flex; align-items: center; gap: 8px; margin: 0; color: var(--text-3); font-size: 12.5px; text-align: center; }
  .notes :global(svg) { flex: none; color: var(--turmeric); }

  @media (max-width: 900px) {
    .job { grid-template-columns: 140px minmax(0, 1fr); }
    .actions { grid-column: 1 / -1; justify-content: flex-start; max-width: none; }
    .steps { grid-template-columns: repeat(2, 1fr); }
  }
  @keyframes rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
