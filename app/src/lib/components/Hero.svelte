<script lang="ts">
  import Icon from "./Icon.svelte";
  import { app } from "../state.svelte";
  import { fmtTime } from "../format";

  let { onOpen }: { onOpen: (url: string) => void } = $props();
  const steps = [
    { icon: "wave", title: "Listens", body: "Finds who speaks when, and what they say." },
    { icon: "sparkle", title: "Translates", body: "Natural spoken Telugu, timed to each line." },
    { icon: "user", title: "Speaks", body: "In a clone of each speaker's own voice." },
  ] as const;
</script>

<section class="hero">
  <div class="headline">
    <h1><span class="te grad-text word">మాట</span></h1>
    <p class="tag">Any YouTube video, spoken in Telugu — in the speaker's own voice.</p>
    <p class="sub">Everything runs on your Mac. Nothing you watch leaves it.</p>
  </div>

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

  {#if app.recents.length}
    <div class="recents">
      <div class="label"><Icon name="clock" size={14} /> Continue watching</div>
      <div class="grid">
        {#each app.recents as r (r.videoId)}
          <button class="card" onclick={() => onOpen(`https://youtu.be/${r.videoId}?t=${Math.floor(r.position)}`)}>
            <div class="thumb" style={`background-image:url(https://i.ytimg.com/vi/${r.videoId}/mqdefault.jpg)`}>
              <span class="pos tabular">{fmtTime(r.position)}</span>
            </div>
            <div class="meta">
              <div class="title">{r.title}</div>
              {#if r.channel}<div class="ch">{r.channel}</div>{/if}
            </div>
          </button>
        {/each}
      </div>
    </div>
  {/if}
</section>

<style>
  .hero { max-width: 980px; margin: 0 auto; padding: 8vh 32px 40px; display: flex; flex-direction: column; gap: 44px; }
  .headline { text-align: center; }
  h1 { margin: 0; line-height: 1; }
  .word { font-size: clamp(88px, 14vw, 168px); font-weight: 700; letter-spacing: -0.02em; filter: drop-shadow(0 10px 40px rgb(242 96 60 / 0.35)); }
  .tag { margin: 18px 0 6px; font-size: 22px; font-weight: 560; letter-spacing: -0.015em; }
  .sub { margin: 0; color: var(--text-2); font-size: 15px; }
  .steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
  .step {
    display: flex; gap: 14px; align-items: flex-start; padding: 18px;
    border-radius: var(--r-lg); background: var(--surface); border: 1px solid var(--border);
    backdrop-filter: var(--blur);
    animation: rise 0.7s var(--ease) both; animation-delay: var(--d);
  }
  .ic { width: 36px; height: 36px; border-radius: 11px; display: grid; place-items: center; color: var(--turmeric); background: rgb(247 183 51 / 0.1); border: 1px solid rgb(247 183 51 / 0.2); flex: none; }
  .t { font-weight: 600; margin-bottom: 2px; }
  .b { color: var(--text-2); font-size: 13px; }
  .recents .label { display: flex; align-items: center; gap: 8px; color: var(--text-2); font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
  .card { text-align: left; padding: 0; border: 1px solid var(--border); background: var(--surface); border-radius: var(--r-md); overflow: hidden; cursor: pointer; transition: transform 0.2s var(--ease), border-color 0.2s; }
  .card:hover { transform: translateY(-2px); border-color: var(--border-strong); }
  .thumb { aspect-ratio: 16/9; background: var(--bg-raised) center/cover; position: relative; }
  .pos { position: absolute; right: 8px; bottom: 8px; font-size: 11px; padding: 2px 6px; border-radius: 6px; background: rgb(0 0 0 / 0.65); color: #fff; }
  .meta { padding: 10px 12px 12px; }
  .title { font-weight: 560; font-size: 13px; display: -webkit-box; -webkit-line-clamp: 2; line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
  .ch { color: var(--text-3); font-size: 12px; margin-top: 2px; }
  @keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
</style>
