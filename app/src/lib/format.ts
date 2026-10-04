/** "2:55:06", "4:05": a video length or position. */
export function fmtTime(s: number): string {
  if (!Number.isFinite(s) || s < 0) s = 0;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const mm = h ? String(m).padStart(2, "0") : String(m);
  return (h ? `${h}:` : "") + `${mm}:${String(sec).padStart(2, "0")}`;
}

/** "40 s", "12 min", "2 h 10 min": a stretch of time as people say it (to the 5 minutes from an hour on). */
export function fmtDuration(s: number): string {
  if (!Number.isFinite(s) || s < 0) s = 0;
  if (s < 59.5) return `${Math.max(1, Math.round(s))} s`;
  if (Math.round(s / 60) < 60) return `${Math.round(s / 60)} min`;
  const five = Math.round(s / 300) * 5;
  const h = Math.floor(five / 60);
  const m = five % 60;
  return m ? `${h} h ${m} min` : `${h} h`;
}

/** Minutes, to the 5 from 20 minutes on. */
const minutes = (s: number): number => Math.max(1, s < 1200 ? Math.round(s / 60) : Math.round(s / 300) * 5);
/** Hours to the half: "3", "6½". */
const hours = (s: number): string => {
  const x = Math.max(0.5, Math.round(s / 1800) / 2);
  return x % 1 ? `${Math.floor(x) || ""}½` : String(x);
};

/** An estimate's range: "35–70 min" (in minutes up to an hour and a half), "50 min – 2 h", "3–6½ h". */
export function fmtSpan(lo: number, hi: number): string {
  if (!Number.isFinite(lo) || lo < 0) lo = 0;
  if (!Number.isFinite(hi) || hi < lo) hi = lo;
  if (hi < 5400) {
    const a = minutes(lo), b = minutes(hi);
    return a === b ? `${b} min` : `${a}–${b} min`;
  }
  if (lo < 3600) return `${minutes(lo)} min – ${hours(hi)} h`;
  const a = hours(lo), b = hours(hi);
  return a === b ? `${b} h` : `${a}–${b} h`;
}

/** "3.1 GB", "850 MB", "12 KB". */
export function fmtBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) n = 0;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  return `${Math.max(1, Math.round(n / 1e3))} KB`;
}

/** "1,900": a count. */
export const fmtCount = (n: number): string => Math.round(n).toLocaleString("en-US");

const SPEAKER_HUES = [24, 330, 190, 145, 265, 48];
export function speakerHue(id: string): number {
  const n = parseInt(id.replace(/\D/g, "") || "1", 10);
  return SPEAKER_HUES[(n - 1) % SPEAKER_HUES.length]!;
}
