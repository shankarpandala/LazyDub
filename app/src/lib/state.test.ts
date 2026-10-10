import { describe, expect, it } from "vitest";
import { AppState, REMOVE_GIVE_UP_MS, REMOVE_RETRY_MS, renderOf } from "./state.svelte";
import { chip, live } from "./jobs";
import { PRIVACY_KEY } from "./claude";
import type { ClientMessage, EngineMessage, JobItem, Output, Render, SpeakersFound, StageRow, VideoInfo } from "./types";

/** A fake engine connection that records what the UI sends. */
function recorder() {
  const sent: ClientMessage[] = [];
  return { sent, send: (m: ClientMessage) => void sent.push(m) };
}

const stage = (key: StageRow["key"], state: StageRow["state"] = "todo", over: Partial<StageRow> = {}): StageRow => ({
  key, label: key, state, done: 0, total: 0, unit: "lines", seconds: 0, eta: null, ...over,
});

const item = (videoId: string, over: Partial<JobItem> = {}): JobItem => ({
  videoId, title: `Video ${videoId}`, channel: "Channel", duration: 600, thumb: null, status: "paused", position: null,
  stopAt: null, settings: { speakers: 2, style: "formal", stopAt: null, speedCap: 1.1, ttsScript: "telugu", presets: [] },
  finalUntil: null, stage: "translate", eta: null, updatedAt: 1, createdAt: 1, bytes: 1000, output: null, slept: null,
  error: null, expired: false, stages: [stage("fetch", "done")], elapsed: 120, coverage: null, report: null, ...over,
});

const render = (videoId: string, over: Partial<Render> = {}): Render => ({
  videoId, status: "running", stage: "fetch", stages: [stage("fetch", "running", { unit: "MB", done: 5, total: 10 })],
  stopAt: null, finalUntil: null, eta: 3600, elapsed: 10, position: null, output: null, slept: null, coverage: null,
  report: null, error: null, ...over,
});

const video = (videoId: string, over: Partial<VideoInfo> = {}): VideoInfo => ({
  videoId, start: 0, title: "A talk", channel: "Someone", duration: 1800, thumbnail: `https://i.ytimg.com/vi/${videoId}/hqdefault.jpg`,
  render: null, estimate: { seconds: [1800, 3600], claudeCalls: [50, 60], ahead: null }, ...over,
});

const hello = (over: Partial<Extract<EngineMessage, { type: "hello" }>> = {}): EngineMessage => ({
  type: "hello", backend: "mock", device: "cpu", demo: true, claude: null, renders: [], render: null,
  settings: { outputDir: "/Users/me/Movies/Maata" }, ...over,
});

describe("hello", () => {
  it("uses the announced voice mode and resets legacy defaults after reconnect", () => {
    const app = new AppState();
    const { send } = recorder();
    expect(app.voiceMode).toBe("cloned");
    expect(app.ttsModel).toBe("Chatterbox Telugu");
    app.receive(hello({ voiceMode: "native", ttsModel: "OmniVoice" }), send);
    expect(app.voiceMode).toBe("native");
    expect(app.ttsModel).toBe("OmniVoice");
    app.receive(hello(), send);
    expect(app.voiceMode).toBe("cloned");
    expect(app.ttsModel).toBe("Chatterbox Telugu");
  });

  it("brings the library, the running job's progress and the settings", () => {
    const app = new AppState();
    const { send } = recorder();
    app.receive(hello({ renders: [item("aaaaaaaaaaa", { status: "running" })], render: render("aaaaaaaaaaa") }), send);
    expect(app.jobs.map((j) => j.videoId)).toEqual(["aaaaaaaaaaa"]);
    expect(app.renders.aaaaaaaaaaa?.eta).toBe(3600);
    expect(app.settings).toEqual({ outputDir: "/Users/me/Movies/Maata" });
    expect(app.view).toBe("library");
  });

  it("opens New dub for a link the app was launched with (?v=, MAATA_OPEN)", () => {
    const app = new AppState();
    const { sent, send } = recorder();
    app.launch = "https://youtu.be/bbbbbbbbbbb";
    app.receive(hello(), send);
    expect(sent).toEqual([{ type: "inspect", url: "https://youtu.be/bbbbbbbbbbb" }]);
    app.receive({ type: "video", ...video("bbbbbbbbbbb") }, send);
    expect(app.view).toBe("new");
    expect(app.video?.videoId).toBe("bbbbbbbbbbb");
    expect(app.launch).toBeNull();
  });
});

describe("video", () => {
  it("opens New dub with the video asked about, and only then", () => {
    const app = new AppState();
    const { send } = recorder();
    app.receive({ type: "video", ...video("ccccccccccc") }, send); // nobody asked: ignored
    expect(app.view).toBe("library");
    expect(app.video).toBeNull();
    app.inspect("  https://youtu.be/ccccccccccc ", send);
    expect(app.inspecting).toBe("https://youtu.be/ccccccccccc");
    app.receive({ type: "video", ...video("ccccccccccc") }, send);
    expect(app.view).toBe("new");
    expect(app.inspecting).toBeNull();
  });

  it("shows the job a video already has, as the library knows it now", () => {
    const app = new AppState();
    const { send } = recorder();
    app.receive(hello({ renders: [item("ddddddddddd", { status: "done" })] }), send);
    app.inspect("ddddddddddd", send);
    app.receive({ type: "video", ...video("ddddddddddd", { render: item("ddddddddddd", { status: "paused" }) }) }, send);
    expect(app.videoJob?.status).toBe("done");
  });

  it("shows only the answer to the link pasted last", () => {
    const app = new AppState();
    const { send } = recorder();
    const a = "https://youtu.be/aaaaaaaaaa1", b = "youtu.be/bbbbbbbbbb2";
    app.inspect(a, send);
    app.inspect(b, send); // corrected before A's answer came
    app.receive({ type: "video", url: a, ...video("aaaaaaaaaa1") }, send);
    expect(app.view).toBe("library");
    expect(app.inspecting).toBe(b);
    app.receive({ type: "error", message: "Couldn't look it up.", retryable: true, url: a }, send); // A's, also late
    expect(app.error).toBeNull();
    expect(app.inspecting).toBe(b);
    app.receive({ type: "video", url: b, ...video("bbbbbbbbbb2") }, send);
    expect(app.view).toBe("new");
    expect(app.video?.videoId).toBe("bbbbbbbbbb2");
    expect(app.inspecting).toBeNull();
  });

  it("a failed inspect shows the error and stays where it was", () => {
    const app = new AppState();
    const { send } = recorder();
    app.inspect("not a link", send);
    app.receive({ type: "error", message: "That isn't a YouTube link.", retryable: false }, send);
    expect(app.inspecting).toBeNull();
    expect(app.error?.message).toBe("That isn't a YouTube link.");
    expect(app.view).toBe("library");
  });
});

describe("dubbing", () => {
  it("sends prepare with the options, and its first render opens the job", () => {
    const app = new AppState();
    const { sent, send } = recorder();
    app.inspect("eeeeeeeeeee", send);
    app.receive({ type: "video", ...video("eeeeeeeeeee") }, send);
    app.dub({ speakers: 2, style: "colloquial", stopAt: 900, speedCap: 1.2, ttsScript: "telugu" }, send);
    expect(sent.at(-1)).toEqual({
      type: "prepare", url: "https://youtu.be/eeeeeeeeeee", speakers: 2, style: "colloquial", stopAt: 900, speedCap: 1.2,
      ttsScript: "telugu",
    });
    app.receive({ type: "render", ...render("eeeeeeeeeee", { status: "queued", position: 1 }) }, send);
    expect(app.view).toBe("job");
    expect(app.openId).toBe("eeeeeeeeeee");
    expect(app.job?.status).toBe("queued");
    expect(sent.at(-1)).toEqual({ type: "renders" }); // a job new to the library: its title and thumbnail come with it
  });

  it("starts New dub from the newest job's options, but for its speakers", () => {
    const app = new AppState();
    app.receive(hello({
      renders: [item("fffffffffff", { createdAt: 5, settings: { speakers: 3, style: "formal", stopAt: 900, speedCap: 1.1, ttsScript: "latin", presets: [] } }),
        item("ggggggggggg", { createdAt: 2 })],
    }), recorder().send);
    expect(app.lastOptions).toEqual({ speakers: "auto", style: "formal", stopAt: 900, speedCap: 1.1, ttsScript: "latin" });
    expect(new AppState().lastOptions).toEqual({ speakers: "auto", style: "colloquial", stopAt: null, speedCap: 1.2, ttsScript: "telugu" });
  });
});

describe("render", () => {
  it("keeps each job's newest progress, and the library's item with it", () => {
    const app = new AppState();
    const { sent, send } = recorder();
    app.receive(hello({ renders: [item("hhhhhhhhhhh")] }), send);
    app.receive({ type: "render", ...render("hhhhhhhhhhh", { stage: "voice_lines", eta: 900, position: null }) }, send);
    expect(app.jobs[0]).toMatchObject({ status: "running", stage: "voice_lines", eta: 900, title: "Video hhhhhhhhhhh" });
    expect(sent).toEqual([]); // known to the library: nothing to ask
    app.openJob("hhhhhhhhhhh");
    expect(app.job?.eta).toBe(900);
    expect(app.jobItem?.title).toBe("Video hhhhhhhhhhh");
  });

  it("shows a job that sends no render from its library item", () => {
    const app = new AppState();
    app.receive(hello({ renders: [item("iiiiiiiiiii", { status: "done", elapsed: 300 })] }), recorder().send);
    app.openJob("iiiiiiiiiii");
    expect(app.job).toEqual(renderOf(app.jobs[0]!));
    expect(app.job).toMatchObject({ status: "done", eta: null, elapsed: 300, stages: [{ key: "fetch", state: "done" }] });
  });

  it("asks for the library once while it waits for it", () => {
    const app = new AppState();
    const { sent, send } = recorder();
    app.receive({ type: "render", ...render("jjjjjjjjjjj") }, send);
    app.receive({ type: "render", ...render("jjjjjjjjjjj") }, send);
    expect(sent).toEqual([{ type: "renders" }]);
    app.receive({ type: "renders", items: [] }, send);
    app.receive({ type: "render", ...render("kkkkkkkkkkk") }, send);
    expect(sent).toEqual([{ type: "renders" }, { type: "renders" }]);
  });
});

describe("renders", () => {
  it("replaces the library, and leaves a job view whose job was removed", () => {
    const app = new AppState();
    const { send } = recorder();
    app.receive({ type: "renders", items: [item("lllllllllll"), item("mmmmmmmmmmm")] }, send);
    app.openJob("lllllllllll");
    app.receive({ type: "renders", items: [item("mmmmmmmmmmm")] }, send);
    expect(app.jobs.map((j) => j.videoId)).toEqual(["mmmmmmmmmmm"]);
    expect(app.view).toBe("library");
  });

  it("takes a job that isn't running from its item: the queue moved up, its MP4 went", () => {
    const app = new AppState();
    const { send } = recorder();
    const out: Output = { path: "/m/A.mp4", bytes: 9, kind: "whole", at: 1, loudness: null, warning: null, bed: true, missing: false };
    app.receive(hello({
      renders: [item("aaaaaaaaaaa", { status: "running" }), item("bbbbbbbbbbb", { status: "queued", position: 1 }),
        item("ccccccccccc", { status: "queued", position: 2 })],
      render: render("aaaaaaaaaaa"),
    }), send);
    app.receive({ type: "render", ...render("ccccccccccc", { status: "queued", position: 2 }) }, send); // its hold
    const chipOf = (id: string) => chip(live(app.jobs.find((j) => j.videoId === id)!, app.renders[id]));
    expect(chipOf("ccccccccccc").text).toBe("Queued · 2nd in line");
    app.receive({ type: "render", ...render("aaaaaaaaaaa", { status: "done", output: out }) }, send);
    // A is done, B runs: the engine sends `renders` only, with C next in line.
    app.receive({ type: "renders", items: [item("bbbbbbbbbbb", { status: "running" }),
      item("ccccccccccc", { status: "queued", position: 1 }), item("aaaaaaaaaaa", { status: "done", output: out })] }, send);
    expect(chipOf("ccccccccccc").text).toBe("Queued · next in line");
    app.openJob("ccccccccccc");
    expect(app.job?.position).toBe(1);
    // A's MP4 is deleted in Finder: the next `renders` says so, over A's last render.
    app.receive({ type: "renders", items: [item("aaaaaaaaaaa", { status: "done", output: { ...out, missing: true } })] }, send);
    expect(chipOf("aaaaaaaaaaa").text).toBe("File moved or deleted");
    app.openJob("aaaaaaaaaaa");
    expect(app.job?.output?.missing).toBe(true);
  });

  it("keeps the render of a running job, which sends newer ones", () => {
    const app = new AppState();
    app.receive(hello({ renders: [item("ddddddddddd", { status: "running" })], render: render("ddddddddddd", { eta: 77 }) }), recorder().send);
    app.receive({ type: "renders", items: [item("ddddddddddd", { status: "running", eta: null })] }, recorder().send);
    expect(app.renders.ddddddddddd?.eta).toBe(77);
  });

  it("removing a running job pauses it first, then asks until the engine has let it go", () => {
    const app = new AppState();
    const { sent, send } = recorder();
    app.receive(hello({ renders: [item("nnnnnnnnnnn", { status: "running" })], render: render("nnnnnnnnnnn") }), send);
    app.remove(app.jobs[0]!, send, 0);
    expect(sent).toEqual([{ type: "pause", videoId: "nnnnnnnnnnn" }]);
    app.retryRemoves(send, REMOVE_RETRY_MS + 1); // still running: nothing
    expect(sent).toHaveLength(1);
    app.receive({ type: "render", ...render("nnnnnnnnnnn", { status: "paused" }) }, send);
    app.retryRemoves(send, 2 * REMOVE_RETRY_MS);
    expect(sent.at(-1)).toEqual({ type: "remove", videoId: "nnnnnnnnnnn" });
    app.receive({ type: "error", message: "Pause it first.", retryable: false }, send); // its run hadn't returned yet
    expect(app.error).toBeNull();
    app.retryRemoves(send, 2 * REMOVE_RETRY_MS + 10); // too soon
    expect(sent).toHaveLength(2);
    app.retryRemoves(send, 3 * REMOVE_RETRY_MS + 10);
    expect(sent).toHaveLength(3);
    app.receive({ type: "renders", items: [] }, send);
    expect(app.removing).toEqual({});
  });

  it("removes a stopped job at once, and gives up on one that never stops", () => {
    const app = new AppState();
    const { sent, send } = recorder();
    app.receive(hello({ renders: [item("ooooooooooo", { status: "done" }), item("ppppppppppp", { status: "waiting" })] }), send);
    app.remove(app.jobs[0]!, send, 0);
    expect(sent).toEqual([{ type: "remove", videoId: "ooooooooooo" }]);
    app.remove(app.jobs[1]!, send, 0);
    app.retryRemoves(send, REMOVE_GIVE_UP_MS + 1);
    expect(app.removing).toEqual({});
    expect(app.error?.message).toMatch(/didn't stop/);
  });
});

describe("speakers_found", () => {
  it("keeps existing presets but never sends a voice switch in native mode", () => {
    const app = new AppState();
    const { sent, send } = recorder();
    const existing = item("rrrrrrrrrrr");
    existing.settings = { ...existing.settings, presets: ["S2"] };
    app.receive(hello({ voiceMode: "native", ttsModel: "OmniVoice", renders: [existing] }), send);
    app.setVoice(existing.videoId, "S2", false, send);
    app.setVoice(existing.videoId, "S1", true, send);
    expect(sent).toEqual([]);
    expect(app.jobs[0]!.settings?.presets).toEqual(["S2"]);
  });

  it("keeps each job's speaker check; the job view shows the open job's", () => {
    const app = new AppState();
    const { send } = recorder();
    const found = {
      videoId: "qqqqqqqqqqq", mode: "auto" as const, fresh: true, freeFor: 1200, bounds: [1, 6] as [number, number],
      speakers: [{ id: "S1", label: "Speaker 1", talkSeconds: 300, share: 0.6, firstAt: 0, turns: 20, activity: [] }], merged: [],
    };
    app.receive({ type: "speakers_found", ...found }, send);
    expect(app.found).toBeNull();
    app.openJob("qqqqqqqqqqq");
    expect(app.found).toEqual(found);
    expect(app.view).toBe("job"); // the check never takes over the screen: it doesn't stop the job (M3)
  });

  it("drops a removed job's check, hides one while the speakers run again, and takes the engine's after hello", () => {
    const app = new AppState();
    const { send } = recorder();
    const found = (videoId: string): SpeakersFound => ({
      videoId, mode: "auto", fresh: false, freeFor: 0, bounds: [1, 6], merged: [],
      speakers: [{ id: "S1", label: "Speaker 1", talkSeconds: 300, share: 1, firstAt: 0, turns: 20, activity: [] }],
    });
    const done = [stage("fetch", "done"), stage("speakers", "done")];
    app.receive(hello({ renders: [item("wwwwwwwwwww", { stages: done }), item("xxxxxxxxxxx", { stages: done })] }), send);
    app.receive({ type: "speakers_found", ...found("wwwwwwwwwww") }, send);
    app.receive({ type: "speakers_found", ...found("xxxxxxxxxxx") }, send);
    app.openJob("xxxxxxxxxxx");
    expect(app.found?.videoId).toBe("xxxxxxxxxxx");
    // Wrong count?: the job diarizes again, and the old check is not of it
    app.receive({ type: "render", ...render("xxxxxxxxxxx", { stages: [stage("fetch", "done"), stage("speakers", "running")] }) }, send);
    expect(app.found).toBeNull();
    app.receive({ type: "render", ...render("xxxxxxxxxxx", { stages: done }) }, send);
    expect(app.found?.videoId).toBe("xxxxxxxxxxx");
    // W removed: dubbing it again shows nothing of the removed job's speakers
    app.receive({ type: "renders", items: [item("xxxxxxxxxxx", { status: "running", stages: done })] }, send);
    expect(Object.keys(app.foundBy)).toEqual(["xxxxxxxxxxx"]);
    app.receive(hello({ renders: [item("xxxxxxxxxxx", { stages: done })] }), send); // reconnected: the engine sends them again
    expect(app.foundBy).toEqual({});
  });

  it("a wrong count and a stock voice go to the engine, and show at once", () => {
    const app = new AppState();
    const { sent, send } = recorder();
    app.receive(hello({ renders: [item("rrrrrrrrrrr")] }), send);
    app.setSpeakers("rrrrrrrrrrr", 3, send);
    app.setVoice("rrrrrrrrrrr", "S2", true, send);
    expect(sent).toEqual([
      { type: "set_speakers", videoId: "rrrrrrrrrrr", speakers: 3 },
      { type: "set_voice", videoId: "rrrrrrrrrrr", speaker: "S2", usePreset: true },
    ]);
    expect(app.jobs[0]!.settings).toMatchObject({ speakers: 3, presets: ["S2"] });
    app.setVoice("rrrrrrrrrrr", "S2", false, send);
    expect(app.jobs[0]!.settings?.presets).toEqual([]);
  });
});

describe("settings", () => {
  it("are the engine's, sent to it and taken from its answer", () => {
    const app = new AppState();
    const { sent, send } = recorder();
    app.saveSettings(" /Volumes/Big/Dubs ", send);
    expect(sent).toEqual([{ type: "settings", outputDir: "/Volumes/Big/Dubs" }]);
    expect(app.settings).toBeNull(); // only the engine's answer changes it
    app.receive({ type: "settings", outputDir: "/Volumes/Big/Dubs" }, send);
    expect(app.settings).toEqual({ outputDir: "/Volumes/Big/Dubs" });
  });
});

describe("continuing a job", () => {
  it("resumes a paused one, takes a preview to the whole video, and dubs an expired one again, with its options", () => {
    const app = new AppState();
    const { sent, send } = recorder();
    app.continueJob(item("sssssssssss", { status: "paused" }), send);
    const preview = item("ttttttttttt", {
      status: "done", stopAt: 900, output: { path: "/x.mp4", bytes: 1, kind: "preview", at: 0, loudness: null, warning: null, missing: false },
      settings: { speakers: 2, style: "formal", stopAt: 900, speedCap: 1.1, ttsScript: "telugu", presets: [] },
    });
    app.continueJob(preview, send);
    app.continueJob(item("uuuuuuuuuuu", { status: "done", expired: true }), send);
    app.continueJob({ ...preview, videoId: "xxxxxxxxxx1", expired: true }, send); // "Dub again": the 15 minutes it was
    app.continueJob({ ...preview, videoId: "xxxxxxxxxx2", expired: true }, send, true); // "Continue to the whole video"
    expect(sent).toEqual([
      { type: "resume", videoId: "sssssssssss" },
      { type: "prepare", url: "https://youtu.be/ttttttttttt", speakers: 2, style: "formal", stopAt: null, speedCap: 1.1, ttsScript: "telugu" },
      { type: "prepare", url: "https://youtu.be/uuuuuuuuuuu", speakers: 2, style: "formal", stopAt: null, speedCap: 1.1, ttsScript: "telugu" },
      { type: "prepare", url: "https://youtu.be/xxxxxxxxxx1", speakers: 2, style: "formal", stopAt: 900, speedCap: 1.1, ttsScript: "telugu" },
      { type: "prepare", url: "https://youtu.be/xxxxxxxxxx2", speakers: 2, style: "formal", stopAt: null, speedCap: 1.1, ttsScript: "telugu" },
    ]);
  });
});

describe("Codex with the existing engine protocol", () => {
  const health = { installed: true, version: "1.0.0", signedIn: true, models: ["gpt-6-luna"], model: "gpt-6-luna",
    problem: null, message: "" };

  it("knows from hello whether the engine translates through Codex, and what is wrong with it already", () => {
    const app = new AppState();
    app.setClaude(null);
    expect(app.claude).toBeNull();
    expect(app.claudeProblem).toBeNull();
    app.setClaude({ ...health, signedIn: false, problem: "not_signed_in", message: "Run codex login." }, 5000);
    expect(app.claude?.version).toBe("1.0.0");
    expect(app.claudeProblem).toEqual({ kind: "not_signed_in", message: "Run codex login.", at: 5000 });
    app.setClaude(health);
    expect(app.claudeProblem).toBeNull();
  });

  it("shows a hold the engine reports, for its job, until a call goes through again", () => {
    const app = new AppState();
    const { send } = recorder();
    app.receive({ type: "claude_error", kind: "usage_limit", message: "limit", limit: "session", resetsAt: 1790300000, retryIn: 600,
      videoId: "vvvvvvvvvvv" }, send, 1000);
    expect(app.claudeProblem).toMatchObject({ kind: "usage_limit", videoId: "vvvvvvvvvvv", resetsAt: 1790300000, at: 1000 });
    app.receive({ type: "claude_ok", videoId: "vvvvvvvvvvv" }, send);
    expect(app.claudeProblem).toBeNull();
  });

  it("requires the new provider notice even after the previous notice was read, then remembers it", () => {
    const store = new Map<string, string>([["maata.claudePrivacySeen", "1"]]);
    const saved = globalThis.localStorage;
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => void store.set(k, v) },
    });
    try {
      const app = new AppState();
      expect(app.privacySeen).toBe(false);
      app.ackPrivacy();
      expect(app.privacySeen).toBe(true);
      expect(store.get(PRIVACY_KEY)).toBe("1");
      expect(store.get("maata.claudePrivacySeen")).toBe("1");
      expect(new AppState().privacySeen).toBe(true);
    } finally {
      Object.defineProperty(globalThis, "localStorage", { configurable: true, value: saved });
    }
  });
});
