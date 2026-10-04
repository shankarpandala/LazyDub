<script lang="ts">
  type Name =
    | "play" | "pause" | "volume" | "stop" | "settings" | "link" | "paste" | "close" | "sparkle" | "wave" | "user" | "cpu"
    | "check" | "alert" | "arrow" | "back" | "clock" | "folder" | "film" | "trash" | "music" | "subtitles" | "redo";
  let { name, size = 18 }: { name: Name; size?: number } = $props();
</script>

<svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  {#if name === "play"}<path d="M7 4.5v15l12-7.5z" fill="currentColor" stroke="none" />
  {:else if name === "pause"}<rect x="6" y="4.5" width="4" height="15" rx="1.2" fill="currentColor" stroke="none" /><rect x="14" y="4.5" width="4" height="15" rx="1.2" fill="currentColor" stroke="none" />
  {:else if name === "stop"}<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none" />
  {:else if name === "volume"}<path d="M4 9v6h4l5 4V5L8 9z" /><path d="M16.5 8.5a5 5 0 0 1 0 7" /><path d="M19 6a8.5 8.5 0 0 1 0 12" />
  {:else if name === "settings"}<circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z" />
  {:else if name === "link"}<path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7" /><path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7" />
  {:else if name === "paste"}<rect x="8" y="3" width="8" height="4" rx="1" /><path d="M16 5h2a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h2" />
  {:else if name === "close"}<path d="M6 6l12 12M18 6 6 18" />
  {:else if name === "sparkle"}<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z" /><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z" />
  {:else if name === "wave"}<path d="M3 12h2M7 8v8M11 5v14M15 9v6M19 7v10M21 12h0" />
  {:else if name === "user"}<circle cx="12" cy="8" r="4" /><path d="M4 21a8 8 0 0 1 16 0" />
  {:else if name === "cpu"}<rect x="6" y="6" width="12" height="12" rx="2" /><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" />
  {:else if name === "check"}<path d="m5 12 5 5L20 7" />
  {:else if name === "alert"}<path d="M12 9v4M12 17h.01" /><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
  {:else if name === "arrow"}<path d="M5 12h14M13 6l6 6-6 6" />
  {:else if name === "back"}<path d="M19 12H5M11 6l-6 6 6 6" />
  {:else if name === "clock"}<circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" />
  {:else if name === "folder"}<path d="M3 7a2 2 0 0 1 2-2h4l2 2.5h8a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
  {:else if name === "film"}<rect x="3" y="4" width="18" height="16" rx="2.5" /><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4" />
  {:else if name === "trash"}<path d="M4 7h16M9 7V4.5h6V7M6.5 7l1 13h9l1-13" />
  {:else if name === "music"}<path d="M9 18V5l11-2v13" /><circle cx="6" cy="18" r="3" /><circle cx="17" cy="16" r="3" />
  {:else if name === "subtitles"}<rect x="3" y="5" width="18" height="14" rx="2.5" /><path d="M7 15h4M13 15h4M7 11h10" />
  {:else if name === "redo"}<path d="M20 8a8.5 8.5 0 1 0 1 6" /><path d="M20 3v5h-5" />
  {/if}
</svg>
