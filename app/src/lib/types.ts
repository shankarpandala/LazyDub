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
  /** Seconds the dub started after the English line (negative: early). Debug HUD only. */
  lag?: number;
  /** Seconds the picture was held for this line, the timeline's rare fallback (0 or absent: none). */
  freeze?: number;
  /** Voice match this line's cloned voice was actually built with, when the engine reports it (never the request). */
  cloneStrength?: CloneStrength;
  /**
   * The take still runs long, and Claude is rephrasing the line in the background: a shorter take may replace this one
   * (the same unit id, sent again) if it comes back in time.
   */
  provisional?: boolean;
  /**
   * false: the engine held this line's audio back, since it is beyond the window near the playhead it keeps in memory
   * and sends (a minute behind to the look-ahead ahead). Ask for it with an `audio` message as the playhead nears it.
   */
  audio?: boolean;
};

export type SpeakerStatus = "found" | "cloning" | "cloned" | "preset";

export type Speaker = {
  id: string;
  label: string;
  voice: "cloned" | "preset";
  status: SpeakerStatus;
  referenceSeconds: number;
  talkSeconds: number;
  usePreset: boolean;
  /**
   * Voice match the engine actually built this speaker's clone with, including a fallback to another match when the
   * requested one couldn't be built. Absent: not reported, and the UI names no match for this voice.
   */
  cloneStrength?: CloneStrength;
};

/** Diarization progress: the speaker pre-pass, then cloning its voices. */
export type SpeakerScan = { phase: "scanning" | "cloning" | "ready"; until: number; duration: number; found: number };

export type VideoInfo = { videoId: string; title: string; channel?: string; duration: number; start: number };

export type Range = [number, number];

/**
 * A stretch of video (from–to, video seconds) the engine flags to the listener. fast_speech: the speech is too fast to
 * dub within the speed-up cap even with the shortest translation, so the dub lags slightly there (spec §3).
 */
export type Notice = { kind: "fast_speech"; from: number; to: number };

export type TranslationStyle = "colloquial" | "formal";

/**
 * What the Telugu voice reads (decision D6): English words in Telugu script (the default), or in Latin letters, for
 * comparing the two by ear. The engine rebuilds the Latin form from the same translation.
 */
export type TtsScript = "telugu" | "latin";

/** Why translation through the Claude CLI can't go on, as the engine classes it (ADR-019). */
export type ClaudeProblemKind =
  | "missing" | "not_signed_in" | "outdated" | "usage_limit" | "transient" | "stalled" | "timeout" | "bad_output" | "failed";

/** The Claude CLI as the engine found it when the UI connected. */
export type ClaudeHealth = {
  installed: boolean;
  version: string | null;
  /** null: the CLI couldn't say. */
  signedIn: boolean | null;
  /** Models this CLI version can run. */
  models: string[];
  /** The model Maata asks for. */
  model: string;
  problem: ClaudeProblemKind | null;
  message: string;
};

/**
 * A Claude call failed in a way translation can't get past on its own. Translation waits `retryIn` s (for a usage
 * limit, until it resets), then tries again; lines already translated keep playing.
 */
export type ClaudeProblem = {
  kind: ClaudeProblemKind;
  message: string;
  /** Which usage limit: session, weekly, opus, sonnet or overage, when known. */
  limit?: string | null;
  /** Epoch seconds the usage limit resets, when known. */
  resetsAt?: number | null;
  retryIn?: number | null;
};

/**
 * Voice match (cloning strength): how closely a cloned voice follows the original speaker's delivery,
 * against how natively it speaks Telugu. The timbre always comes from the speaker.
 */
export type CloneStrength = "closest" | "balanced" | "natural";

export type EngineMessage =
  /** claude: null when the engine never calls Claude (the demo engine). */
  | { type: "hello"; backend: string; device: string; demo: boolean; claude?: ClaudeHealth | null }
  | ({ type: "video" } & VideoInfo)
  | { type: "status"; stage: string; message: string; at: number | null }
  | ({ type: "unit" } & DubUnit)
  | { type: "unit_skipped"; id: number; start: number; end: number }
  /**
   * until: end of the contiguous dub from the engine's playhead; ranges: merged dubbed video ranges;
   * throughput: engine speed (× realtime EWMA, 0 = unknown); targetLead: lead the engine recommends before playing.
   */
  | { type: "ready"; until: number; ranges: Range[]; throughput: number; targetLead: number }
  | ({ type: "speaker_scan" } & SpeakerScan)
  | { type: "speakers"; speakers: Speaker[] }
  | { type: "stage_time"; stage: string; seconds: number; window: number }
  | ({ type: "notice" } & Notice)
  | ({ type: "claude_error" } & ClaudeProblem)
  /** A Claude call went through again after a claude_error. */
  | { type: "claude_ok" }
  | { type: "error"; message: string; retryable: boolean };

export type ClientMessage =
  /**
   * speedCap: most a line may be sped up to fit its slot (1.0–1.25); allowFreeze: the picture may be held briefly,
   * rarely, when a line still can't fit. The video is never slowed down.
   */
  | {
      type: "open"; url: string; style: TranslationStyle; lookahead: number;
      cloneStrength: CloneStrength; speedCap: number; allowFreeze: boolean; ttsScript: TtsScript;
    }
  | { type: "player"; rates: number[] }
  | { type: "seek"; time: number }
  /** playing: false while the video waits (paused, or for the dub): a slow engine uses that time to work further ahead. */
  | { type: "playhead"; time: number; playing?: boolean }
  /** Send the audio of these lines, which the UI doesn't hold (held back, or evicted). */
  | { type: "audio"; ids: number[] }
  /** "Prepare the whole video": dub to the end, whatever the look-ahead. */
  | { type: "prepare"; whole: boolean }
  | { type: "speed"; speed: number }
  | { type: "speaker_preset"; speaker: string; usePreset: boolean }
  /** Playback state for the engine log while the user wants to play: the audio output's state, gain and what is held. */
  | {
      type: "diag"; audio: string; rate: number; gain: number; buffers: number; live: number; units: number;
      volume: number; lead: number; waiting: boolean; missing: number; time: number;
    }
  | { type: "close" };
