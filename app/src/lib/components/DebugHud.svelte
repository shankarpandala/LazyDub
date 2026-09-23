<script lang="ts">
  import { app } from "../state.svelte";
  import { percentile } from "../format";

  let { errors, scheduled }: { errors: number[]; scheduled: number } = $props();
  const p50 = $derived(percentile(errors.map(Math.abs), 50));
  const p95 = $derived(percentile(errors.map(Math.abs), 95));
  const lastWin = $derived(app.windowSeconds.at(-1));
  const fmt = (x: number | null | undefined, d = 0) => (x == null ? "—" : x.toFixed(d));
</script>

<div class="hud tabular" aria-label="Debug HUD">
  <div><span>lead</span>{fmt(app.lead, 1)} s</div>
  <div><span>start err p50</span>{fmt(p50)} ms</div>
  <div><span>p95</span>{fmt(p95)} ms</div>
  <div><span>n</span>{errors.length}</div>
  <div><span>live</span>{scheduled}</div>
  <div><span>window</span>{fmt(lastWin, 1)} s</div>
  <div><span>engine</span>{app.backend || "—"}</div>
</div>

<style>
  .hud {
    position: fixed; left: 16px; bottom: 16px; z-index: 15; display: grid; grid-template-columns: repeat(7, auto); gap: 14px;
    padding: 8px 12px; border-radius: 10px; font: 500 11px ui-monospace, "SF Mono", monospace;
    background: rgb(0 0 0 / 0.72); color: #d7ffe9; border: 1px solid rgb(61 220 151 / 0.25);
  }
  span { color: rgb(215 255 233 / 0.55); margin-right: 5px; }
</style>
