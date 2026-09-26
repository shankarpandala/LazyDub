<script lang="ts">
  import Icon from "./Icon.svelte";
  import { app } from "../state.svelte";
  import { PRIVACY_NOTICE, claudeNotice } from "../claude";

  // The countdown in "Trying again in 40 s" moves while the banner is up.
  let now = $state(Date.now());
  $effect(() => {
    if (!app.claudeProblem) return;
    const id = setInterval(() => (now = Date.now()), 1000);
    return () => clearInterval(id);
  });

  const problem = $derived(app.claudeProblem ? claudeNotice(app.claudeProblem, app.claudeProblem.at, now) : null);
  // Shown once, before the first video goes out to Claude; never on the demo engine, which doesn't use it.
  const privacy = $derived(!!app.claude && !app.privacySeen);
</script>

{#if problem || privacy}
  <div class="banners">
    {#if problem}
      <div class="banner warn" role="alert">
        <span class="ic"><Icon name="alert" size={16} /></span>
        <div class="text">
          <div class="t">{problem.title}</div>
          <div class="d">
            {problem.detail}
            {#if problem.command}<span class="run">Run <code>{problem.command}</code> in a terminal.</span>{/if}
          </div>
        </div>
      </div>
    {/if}
    {#if privacy}
      <div class="banner info" role="note" aria-labelledby="privacy-title">
        <span class="ic"><Icon name="sparkle" size={16} /></span>
        <div class="text">
          <div class="t" id="privacy-title">{PRIVACY_NOTICE.title}</div>
          <div class="d">{PRIVACY_NOTICE.detail}</div>
        </div>
        <button class="ok" onclick={() => app.ackPrivacy()}>Got it</button>
      </div>
    {/if}
  </div>
{/if}

<style>
  /* Lined up with the watch view below it (App.svelte .watch). */
  .banners { display: flex; flex-direction: column; gap: 8px; max-width: 1500px; margin: 0 auto; padding: 10px 28px 0; }
  .banner {
    display: flex; align-items: flex-start; gap: 12px; padding: 12px 16px;
    border-radius: var(--r-lg); background: var(--surface); border: 1px solid var(--border); backdrop-filter: var(--blur);
    animation: rise 0.3s var(--ease);
  }
  .warn { border-color: rgb(247 183 51 / 0.4); background: rgb(247 183 51 / 0.08); }
  .warn .ic { color: var(--warn-ink); }
  .info .ic { color: var(--vermilion); }
  .ic { flex: none; margin-top: 1px; }
  .text { flex: 1; min-width: 0; }
  .t { font-weight: 620; }
  .d { color: var(--text-2); font-size: 12.5px; margin-top: 2px; user-select: text; -webkit-user-select: text; }
  .run { display: inline; margin-left: 2px; }
  code { font: 12px ui-monospace, monospace; background: var(--surface-strong); padding: 1px 6px; border-radius: 6px; }
  .ok {
    flex: none; align-self: center; border: 0; cursor: pointer; padding: 7px 14px; border-radius: 10px; font-weight: 600;
    background: var(--surface-strong); color: var(--text);
  }
  .ok:hover { background: var(--surface-hover); }
  @keyframes rise { from { opacity: 0; transform: translateY(-4px); } }
  @media (prefers-reduced-motion: reduce) { .banner { animation: none; } }
</style>
