import { describe, expect, it } from "vitest";
import { NOTICE_LEAD, addNotice, noticeAt, noticeText } from "./notices";
import type { Notice } from "./types";

const fast = (from: number, to: number): Notice => ({ kind: "fast_speech", from, to });

describe("addNotice", () => {
  it("keeps stretches in video order", () => {
    expect(addNotice([fast(100, 140)], fast(20, 60))).toEqual([fast(20, 60), fast(100, 140)]);
  });
  it("merges overlapping and touching stretches", () => {
    expect(addNotice([fast(20, 60)], fast(50, 90))).toEqual([fast(20, 90)]);
    expect(addNotice([fast(20, 60)], fast(60, 90))).toEqual([fast(20, 90)]);
    expect(addNotice([fast(20, 60), fast(100, 140)], fast(55, 105))).toEqual([fast(20, 140)]);
  });
  it("is unchanged by a repeat of the same stretch", () => {
    expect(addNotice([fast(20, 60)], fast(20, 60))).toEqual([fast(20, 60)]);
  });
  it("drops a notice it can't place", () => {
    expect(addNotice([], fast(60, 20))).toEqual([]);
    expect(addNotice([], fast(30, 30))).toEqual([]);
    expect(addNotice([], fast(Number.NaN, 30))).toEqual([]);
    expect(addNotice([], { kind: "loud_music", from: 0, to: 30 } as unknown as Notice)).toEqual([]);
  });
  it("never changes the list it was given", () => {
    const list = [fast(20, 60)];
    addNotice(list, fast(50, 90));
    expect(list).toEqual([fast(20, 60)]);
  });
});

describe("noticeAt", () => {
  const list = [fast(20, 60), fast(100, 140)];
  it("covers a stretch from a moment before it to its end", () => {
    expect(noticeAt(list, 20 - NOTICE_LEAD)).toEqual(fast(20, 60));
    expect(noticeAt(list, 45)).toEqual(fast(20, 60));
    expect(noticeAt(list, 120)).toEqual(fast(100, 140));
  });
  it("is null outside every stretch", () => {
    expect(noticeAt(list, 10)).toBeNull();
    expect(noticeAt(list, 60)).toBeNull();
    expect(noticeAt(list, 80)).toBeNull();
    expect(noticeAt([], 45)).toBeNull();
  });
});

describe("noticeText", () => {
  it("explains the lag in plain words, as a warning, not an error", () => {
    expect(noticeText(fast(20, 60))).toBe("Fast speech here · the dub lags slightly");
  });
});
