export function fmtTime(s: number): string {
  if (!Number.isFinite(s) || s < 0) s = 0;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const mm = h ? String(m).padStart(2, "0") : String(m);
  return (h ? `${h}:` : "") + `${mm}:${String(sec).padStart(2, "0")}`;
}

export function percentile(xs: number[], p: number): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))]!;
}

/** Merge [start, end] intervals (dub-ready ranges on the scrubber). */
export function mergeRanges(rs: [number, number][], gap = 1.5): [number, number][] {
  const s = [...rs].sort((a, b) => a[0] - b[0]);
  const out: [number, number][] = [];
  for (const r of s) {
    const last = out[out.length - 1];
    if (last && r[0] <= last[1] + gap) last[1] = Math.max(last[1], r[1]);
    else out.push([r[0], r[1]]);
  }
  return out;
}

const SPEAKER_HUES = [24, 330, 190, 145, 265, 48];
export function speakerHue(id: string): number {
  const n = parseInt(id.replace(/\D/g, "") || "1", 10);
  return SPEAKER_HUES[(n - 1) % SPEAKER_HUES.length]!;
}
