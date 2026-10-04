import type { ClientMessage, EngineMessage } from "./types";

export type AudioFrame = { id: number; sampleRate: number; samples: Float32Array };

type Handlers = {
  message: (m: EngineMessage) => void;
  audio: (a: AudioFrame) => void;
  connection: (state: "connecting" | "open" | "closed") => void;
};

declare global {
  interface Window {
    __MAATA__?: { wsUrl: string };
  }
}

/** Engine URL: injected by the Tauri shell, else same-origin with ?token= (engine-served UI, ADR-006). */
export function engineUrl(loc: Location = location): string | null {
  if (window.__MAATA__?.wsUrl) return window.__MAATA__.wsUrl;
  const q = new URLSearchParams(loc.search);
  const token = q.get("token");
  if (!token) return null;
  const host = q.get("engine") ?? loc.host;
  return `ws://${host}/ws?token=${encodeURIComponent(token)}`;
}

/**
 * A path the engine serves (a cached thumbnail, `/thumb/<id>?token=`) as the page can load it: same-origin for the
 * engine-served UI; on the engine's host for a UI served elsewhere with ?engine= (development).
 */
export function engineAsset(path: string, loc: Location = location): string {
  const host = new URLSearchParams(loc.search).get("engine");
  return host && path.startsWith("/") ? `http://${host}${path}` : path;
}

/** Frame id of a speaker's Hear voice sample: this plus the speaker's index (S1: 0), the one frame the engine sends. */
export const SAMPLE_ID = 0xffffff00;

/** The speaker ("S2") whose Hear voice sample a frame is, or null for any other frame. */
export function sampleSpeaker(id: number): string | null {
  const k = id - SAMPLE_ID;
  return k >= 0 && k < 255 ? `S${k + 1}` : null;
}

/** Binary frame (ADR-007): u32 id, u32 sample rate, u32 sample count, u32 reserved, then float32 PCM. */
export function decodeAudioFrame(buf: ArrayBuffer): AudioFrame {
  const h = new DataView(buf, 0, 16);
  const id = h.getUint32(0, true);
  const sampleRate = h.getUint32(4, true);
  const n = h.getUint32(8, true);
  return { id, sampleRate, samples: new Float32Array(buf, 16, n) };
}

export class EngineClient {
  private ws: WebSocket | null = null;
  private retry = 0;
  private closed = false;

  constructor(private url: string, private on: Handlers) {}

  connect(): void {
    this.on.connection("connecting");
    const ws = new WebSocket(this.url);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.retry = 0;
      this.on.connection("open");
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") this.on.message(JSON.parse(ev.data) as EngineMessage);
      else this.on.audio(decodeAudioFrame(ev.data as ArrayBuffer));
    };
    ws.onclose = () => {
      this.on.connection("closed");
      if (!this.closed) setTimeout(() => this.connect(), Math.min(4000, 250 * 2 ** this.retry++));
    };
    this.ws = ws;
  }

  send(msg: ClientMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  close(): void {
    this.closed = true;
    this.ws?.close();
  }
}
