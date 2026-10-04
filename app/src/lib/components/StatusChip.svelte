<script lang="ts">
  import type { Chip } from "../jobs";

  let { chip }: { chip: Chip } = $props();
</script>

<span class={`chip ${chip.tone}`} role="status"><span class="pulse" aria-hidden="true"></span>{chip.text}</span>

<style>
  .chip {
    display: inline-flex; align-items: center; gap: 7px; max-width: 100%;
    height: 24px; padding: 0 10px; border-radius: 999px; font-size: 12px; font-weight: 560;
    background: var(--surface); border: 1px solid var(--border); color: var(--text-2);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .pulse { width: 7px; height: 7px; border-radius: 50%; background: var(--text-3); flex: none; }
  .run { color: var(--text); border-color: rgb(242 96 60 / 0.35); background: rgb(242 96 60 / 0.08); }
  .run .pulse { background: var(--vermilion); box-shadow: 0 0 0 0 rgb(242 96 60 / 0.6); animation: beat 1.6s ease-out infinite; }
  .wait .pulse { background: var(--turmeric); }
  .warn { color: var(--warn-ink); border-color: rgb(247 183 51 / 0.4); background: rgb(247 183 51 / 0.08); }
  .warn .pulse { background: var(--warn); }
  .ok { color: var(--text); }
  .ok .pulse { background: var(--ok); }
  .danger { color: var(--danger); border-color: rgb(255 93 93 / 0.35); background: rgb(255 93 93 / 0.08); }
  .danger .pulse { background: var(--danger); }
  @keyframes beat { 70% { box-shadow: 0 0 0 6px rgb(242 96 60 / 0); } 100% { box-shadow: 0 0 0 0 rgb(242 96 60 / 0); } }
</style>
