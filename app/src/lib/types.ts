export type Edit = { kind: "slow" | "freeze"; at: number; span: number; rate: number; added: number };

export type DubUnit = {
  id: number;
  speaker: string;
  start: number;
  end: number;
  budget: number;
  audioRate: number;
  audioWall: number;
  voice: "cloned" | "preset";
  source: string;
  telugu: string;
  units: number;
  edits: Edit[];
};

export type Speaker = { id: string; label: string; voice: "cloned" | "preset"; referenceSeconds: number; usePreset: boolean };

export type VideoInfo = { videoId: string; title: string; channel?: string; duration: number; start: number };

export type EngineMessage =
  | { type: "hello"; backend: string; device: string; demo: boolean }
  | ({ type: "video" } & VideoInfo)
  | { type: "status"; stage: string; message: string; at: number | null }
  | ({ type: "unit" } & DubUnit)
  | { type: "unit_skipped"; id: number; start: number; end: number }
  | { type: "ready"; until: number }
  | { type: "speakers"; speakers: Speaker[] }
  | { type: "stage_time"; stage: string; seconds: number; window: number }
  | { type: "error"; message: string; retryable: boolean };
