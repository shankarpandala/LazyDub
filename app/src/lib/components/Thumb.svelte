<script lang="ts">
  import { fmtTime } from "../format";

  /** A video's thumbnail, or the Telugu mark on the accent when there is none (the demo) or it doesn't load. */
  let { src, duration = 0, title = "" }: { src: string | null; duration?: number; title?: string } = $props();
  let broken = $state(false);
  $effect(() => {
    void src;
    broken = false;
  });
</script>

<div class="thumb" class:empty={!src || broken}>
  {#if src && !broken}
    <img {src} alt="" loading="lazy" decoding="async" onerror={() => (broken = true)} />
  {:else}
    <span class="te mark" aria-hidden="true">మా</span>
  {/if}
  {#if duration > 0}<span class="len tabular" aria-label={`${title} is ${fmtTime(duration)} long`}>{fmtTime(duration)}</span>{/if}
</div>

<style>
  .thumb { position: relative; aspect-ratio: 16 / 9; border-radius: var(--r-md); overflow: hidden; background: var(--bg-raised); }
  img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .empty { background: linear-gradient(135deg, rgb(247 183 51 / 0.22), rgb(242 96 60 / 0.18) 50%, rgb(216 51 107 / 0.22)); display: grid; place-items: center; }
  .mark { font-size: 28px; font-weight: 700; color: rgb(255 255 255 / 0.75); }
  :global([data-theme="light"]) .mark { color: rgb(27 22 36 / 0.55); }
  .len { position: absolute; right: 6px; bottom: 6px; font-size: 11px; font-weight: 600; padding: 1px 6px; border-radius: 6px; background: rgb(0 0 0 / 0.7); color: #fff; }
</style>
