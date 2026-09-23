<script lang="ts">
  import { app } from "../state.svelte";
  import { fmtTime, mergeRanges } from "../format";

  let { onSeek }: { onSeek: (t: number) => void } = $props();
  let track: HTMLDivElement;
  let hover = $state<number | null>(null);
  let dragging = $state(false);

  const dur = $derived(Math.max(app.duration || app.video?.duration || 0, 1));
  // Dub coverage: lines plus the silences between them, up to the engine's ready horizon.
  const ranges = $derived(mergeRanges(app.units.map((u) => [u.start, Math.min(u.start + u.budget + 4, app.readyUntil)] as [number, number]), 4));
  const pct = (t: number) => `${Math.min(100, Math.max(0, (t / dur) * 100))}%`;

  function timeAt(clientX: number) {
    const r = track.getBoundingClientRect();
    return Math.min(dur, Math.max(0, ((clientX - r.left) / r.width) * dur));
  }
  function down(e: PointerEvent) {
    dragging = true;
    track.setPointerCapture(e.pointerId);
    onSeek(timeAt(e.clientX));
  }
  function move(e: PointerEvent) {
    hover = timeAt(e.clientX);
    if (dragging) onSeek(hover);
  }
  function key(e: KeyboardEvent) {
    const step = e.shiftKey ? 30 : 5;
    if (e.key === "ArrowRight") { onSeek(Math.min(dur, app.time + step)); e.preventDefault(); }
    if (e.key === "ArrowLeft") { onSeek(Math.max(0, app.time - step)); e.preventDefault(); }
  }
</script>

<div
  class="scrub"
  class:active={dragging || hover !== null}
  bind:this={track}
  role="slider"
  tabindex="0"
  aria-label="Seek"
  aria-valuemin="0"
  aria-valuemax={Math.round(dur)}
  aria-valuenow={Math.round(app.time)}
  aria-valuetext={`${fmtTime(app.time)} of ${fmtTime(dur)}`}
  onpointerdown={down}
  onpointermove={move}
  onpointerup={() => (dragging = false)}
  onpointerleave={() => { if (!dragging) hover = null; }}
  onkeydown={key}
>
  <div class="rail">
    {#each ranges as [a, b] (a)}
      <div class="ready" style={`left:${pct(a)};width:calc(${pct(b)} - ${pct(a)})`}></div>
    {/each}
    <div class="played" style={`width:${pct(app.time)}`}></div>
    {#each app.skipped as s (s.id)}
      <div class="skip" style={`left:${pct(s.start)}`} title="Line skipped"></div>
    {/each}
    {#each app.units.flatMap((u) => u.edits.filter((e) => e.kind === "freeze")) as e (e.at)}
      <div class="freeze" style={`left:${pct(e.at)}`}></div>
    {/each}
  </div>
  <div class="knob" style={`left:${pct(app.time)}`}></div>
  {#if hover !== null}
    <div class="tip tabular" style={`left:${pct(hover)}`}>{fmtTime(hover)}</div>
  {/if}
</div>

<style>
  .scrub { position: relative; height: 22px; display: flex; align-items: center; cursor: pointer; touch-action: none; }
  .rail { position: relative; width: 100%; height: 4px; border-radius: 99px; background: rgb(255 255 255 / 0.12); overflow: hidden; transition: height 0.15s var(--ease); }
  .scrub.active .rail { height: 6px; }
  .ready { position: absolute; top: 0; bottom: 0; background: linear-gradient(90deg, rgb(247 183 51 / 0.35), rgb(242 96 60 / 0.35)); }
  .played { position: absolute; left: 0; top: 0; bottom: 0; background: var(--accent-grad); border-radius: 99px; }
  .skip { position: absolute; top: 0; bottom: 0; width: 3px; background: var(--danger); }
  .freeze { position: absolute; top: 0; bottom: 0; width: 2px; background: rgb(255 255 255 / 0.8); }
  .knob {
    position: absolute; top: 50%; width: 14px; height: 14px; margin-left: -7px; margin-top: -7px; border-radius: 50%;
    background: #fff; box-shadow: 0 0 0 4px rgb(242 96 60 / 0.35), 0 2px 8px rgb(0 0 0 / 0.5);
    transform: scale(0); transition: transform 0.15s var(--ease);
  }
  .scrub.active .knob, .scrub:focus-visible .knob { transform: scale(1); }
  .tip {
    position: absolute; bottom: 24px; transform: translateX(-50%);
    font-size: 11px; font-weight: 600; padding: 3px 7px; border-radius: 6px; background: rgb(0 0 0 / 0.8); color: #fff; pointer-events: none;
  }
</style>
