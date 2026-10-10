"""Global speaker registry (spec §6.2), built for the streaming player's speaker pre-pass (gone with ADR-021).

The pre-pass diarized the audio in long, overlapping blocks. Each block came back with its own
labels (pyannote's SPEAKER_00, …), so one person could be SPEAKER_01 in one block and SPEAKER_00 in
the next. This registry links block labels to stable ids (S1, S2, …) by speaker-embedding
similarity, so a speaker who is silent for a while comes back with the same id. The speech a label
shares with known turns in the block overlap (the same audio, diarized twice) backs the embedding up,
and stands in for it when there is none. Labels with too little speech to trust (a laugh, a split-off
fragment) only follow a known speaker. The registry keeps one non-duplicated timeline of turns and
picks clean reference audio for cloning: one contiguous span for the timbre prompt (best_span), and
many clean clips to average a speaker embedding over (clean_clips).

The offline render (OFFLINE-RENDER §2.3) diarizes the whole file at once instead: one `add_block` registers it, `settle`
folds over-split speakers back, `match_ids` keeps a re-run's ids, and `activity` gives the UI's per-speaker strip.

Pure logic (numpy only, ADR-005). Times are absolute video seconds.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from .types import SpeakerTurn

_EPS = 1e-6
_JOIN = 1e-3               # s: coverage spans / same-speaker pieces closer than this are joined, and the
                           #    slack of coverage queries (block ends are whole 16 kHz samples, so can fall short)
_MIN_PIECE = 0.02          # s: slivers left by clipping at a coverage boundary are dropped
_MIN_LINK_OVERLAP = 0.2    # s of shared exclusive speech needed to link a label by overlap
_MIN_OWN = 2.0             # s of exclusive speech in its block a label needs to claim or found an id
_VOUCH = 2.0               # s of shared exclusive speech that backs up an embedding match…
_VOUCH_BONUS = 0.2         # …by this much cosine
_TALK_TIE = 0.1            # cosine: labels contending for one id get up to this much priority for speech…
_TALK_SAT = 10.0           # s: …saturating here
_MIN_WEIGHT = 0.1          # s: floor on the speech behind a block centroid
_NEAR = 0.5                # s: speaker_at falls back to the nearest turn within this distance
_LENGTH_SAT = 8.0          # s: a reference clip stops scoring for length here
_ISOLATION_CAP = 2.0       # s: a clip this far from other speakers is fully isolated
_ISOLATION_WEIGHT = 0.3
_BRIDGE = 0.6              # s: a pause this short between one speaker's turns keeps their speech one contiguous span
_CLEAN_MAX = 10.0          # s: clean_clips cuts longer spans into pieces no longer than this
_SCORE_TOP = 8             # best_span asks the caller's score() about this many candidates at a time…
_SCORE_TIE = 0.05          # …taking candidates whose built-in scores are this close as ties, spread over the timeline
# Settling a whole-file diarization (OFFLINE-RENDER §2.3, decision M2): in auto mode a speaker with less exclusive talk
# than max(MIN_TALK, MIN_SHARE x everyone's) is merged into the nearest one, and so is either of two speakers whose
# centroids are at least SAME_COS alike. The floor is never more than FLOOR_CAP x everyone's talk, so a clip with under
# 5 minutes of speech keeps a second speaker who talks less than MIN_TALK (from 5 minutes on, the cap changes nothing).
MIN_TALK = 30.0            # s
MIN_SHARE = 0.01
SAME_COS = 0.85
FLOOR_CAP = 0.1
_ADJACENT = 1.0            # s either side of a speaker's turn in which another's speech counts as next to theirs
VOICE_PROFILE_POLICY = "source-pitch-v1"
VOICE_PROFILE_SECONDS = 12.0


def infer_voice_profile(clips: list[np.ndarray], sample_rate: int = 16_000) -> dict:
    """Conservative vocal-range matching, not a person's gender identity.

    Analyse at most twelve seconds of clean diarized speech with a YIN-style normalized difference function.
    Low/high stable pitch may select the corresponding male/female synthesis profile; overlapping ranges, too
    little periodic speech, noise and silence remain unresolved for the user to choose. No text/name inference.
    The voiced fraction screens periodic versus unvoiced frames; it is not a calibrated confidence probability.
    The caller must supply speech clips: pitch alone cannot reliably distinguish a musical note from a voice.
    """
    if sample_rate != 16_000:
        raise ValueError("Voice profile analysis expects 16 kHz source audio")
    pitches, usable, sampled = [], 0, 0
    limit = round(VOICE_PROFILE_SECONDS * sample_rate)
    sr, frame, hop, lo, hi = 8_000, 400, 160, 20, 123  # 50 ms; 20 ms hop; roughly 65–400 Hz
    for clip in clips:
        x = np.asarray(clip, np.float64).reshape(-1)[:max(0, limit - sampled)]
        sampled += len(x)
        if len(x) < 2 * frame or not np.isfinite(x).all():
            continue
        x = x[:len(x) // 2 * 2].reshape(-1, 2).mean(axis=1)
        frames = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop].copy()
        frames -= frames.mean(axis=1, keepdims=True)
        rms = np.sqrt(np.mean(frames * frames, axis=1))
        frames = frames[rms >= max(1e-4, float(rms.max()) * 0.1)]
        usable += len(frames)
        if not len(frames):
            continue
        width = frame - hi
        diff = np.zeros((len(frames), hi + 1))
        for lag in range(1, hi + 1):
            delta = frames[:, :width] - frames[:, lag:lag + width]
            diff[:, lag] = np.sum(delta * delta, axis=1)
        running = np.cumsum(diff[:, 1:], axis=1)
        norm = np.ones_like(diff)
        norm[:, 1:] = diff[:, 1:] * np.arange(1, hi + 1) / np.maximum(running, 1e-20)
        for row in norm:
            # First sufficiently periodic local minimum avoids choosing an octave below a strong fundamental.
            candidates = np.flatnonzero((row[lo:hi] < 0.15) & (row[lo:hi] <= row[lo - 1:hi - 1])
                                        & (row[lo:hi] <= row[lo + 1:hi + 1]))
            if not len(candidates):
                continue
            lag = int(candidates[0]) + lo
            denom = row[lag - 1] - 2 * row[lag] + row[lag + 1]
            offset = 0.5 * (row[lag - 1] - row[lag + 1]) / denom if denom > 0 else 0.0
            pitches.append(sr / (lag + float(np.clip(offset, -0.5, 0.5))))
        if sampled >= limit:
            break
    voiced = len(pitches) * hop / sr
    fraction = len(pitches) / usable if usable else 0.0
    pitch = float(np.median(pitches)) if pitches else None
    profile = None
    if voiced >= 2.0 and fraction >= 0.2:
        p = np.asarray(pitches)
        if np.mean((p >= 70) & (p <= 160)) >= 0.8:
            profile = "male"
        elif np.mean((p >= 190) & (p <= 350)) >= 0.8:
            profile = "female"
    return {"policy": VOICE_PROFILE_POLICY, "profile": profile, "pitchHz": round(pitch, 1) if pitch else None,
            "voicedSeconds": round(voiced, 2), "voicedFraction": round(fraction, 3),
            "periodicFrames": len(pitches), "analysedFrames": usable,
            "sampleSeconds": round(sampled / sample_rate, 2)}


@dataclass(frozen=True, slots=True)
class DiarBlock:
    """One diarized audio block. Times are absolute; labels are local to the block."""

    start: float                       # absolute video time the diarized audio block starts
    end: float
    turns: list[SpeakerTurn]           # may overlap (crosstalk)
    exclusive: list[SpeakerTurn]       # non-overlapping (pyannote exclusive diarization)
    centroids: dict[str, np.ndarray]   # label -> speaker embedding (may be missing/NaN for tiny speakers)
    step: float | None = None          # whole-file diarization: the segmentation step used, as a share of its window
    embeddings: int | None = None      # whole-file diarization: the embeddings its clustering kept (the guard's count)


@dataclass
class GlobalSpeaker:
    id: str                    # "S1", "S2", …: never reused
    label: str                 # "Speaker 1", … (renamable)
    centroid: np.ndarray | None  # unit vector, talk-time-weighted mean of block embeddings
    talk_seconds: float        # exclusive speech registered so far
    first_at: float


class _Track:
    """Turns sorted by start, with touching same-speaker pieces joined, for bisect range queries."""

    def __init__(self) -> None:
        self.turns: list[SpeakerTurn] = []
        self._starts: list[float] = []
        self._max_len = 0.0

    def set(self, turns: list[SpeakerTurn]) -> None:
        out: list[SpeakerTurn] = []
        last: dict[str, int] = {}
        for t in sorted(turns, key=lambda t: (t.start, t.end, t.speaker)):
            i = last.get(t.speaker)
            if i is not None and t.start <= out[i].end + _JOIN:
                if t.end > out[i].end:
                    out[i] = SpeakerTurn(t.speaker, out[i].start, t.end)
                continue
            last[t.speaker] = len(out)
            out.append(t)
        self.turns = out
        self._starts = [t.start for t in out]
        self._max_len = max((t.end - t.start for t in out), default=0.0)

    def extend(self, new: list[SpeakerTurn]) -> None:
        if new:
            self.set(self.turns + new)

    def near(self, a: float, b: float) -> list[SpeakerTurn]:
        """Turns touching the closed range [a, b], in start order."""
        lo = bisect_left(self._starts, a - self._max_len - _EPS)
        hi = bisect_right(self._starts, b)
        return [t for t in self.turns[lo:hi] if t.end >= a]


class SpeakerRegistry:
    def __init__(self, link_threshold: float = 0.5) -> None:
        self.link_threshold = link_threshold
        self.speakers: dict[str, GlobalSpeaker] = {}
        self._all = _Track()                            # overlapping turns, crosstalk included
        self._ex = _Track()                             # exclusive turns: one speaker at a time
        self._stray = _Track()                          # speech of short labels that matched nobody
        self._coverage: list[tuple[float, float]] = []  # diarized spans, sorted and merged
        self._weight: dict[str, float] = {}             # seconds of speech behind each centroid
        self._issued = 0                                # highest id number ever handed out

    # -- registration ---------------------------------------------------------------------------

    def add_block(self, block: DiarBlock) -> dict[str, str]:
        """Register a diarized block and return its local label -> global id mapping.

        Only the part of the block not diarized before is taken: its turns are clipped at the earlier
        coverage, so an overlap never doubles up.

        Labels with at least 2 s of exclusive speech in the block claim ids one-to-one, best match
        first. A match is the embedding cosine, plus 0.2 when the label shares most of its overlap
        speech (>= 2 s of it) with that speaker; among labels after the same id, more speech breaks
        near-ties. A label with no usable embedding links by shared exclusive speech (so does one whose
        embedding matched nobody, but only to a speaker with no embedding yet). With no embeddings on
        either side (the single-speaker fallback), a block's lone label continues the lone speaker even
        across a seek gap. Unlinked labels become new speakers in order of their first new speech; one
        that brings no new speech is left out of the mapping.

        A shorter label is too little speech to trust (a laugh, a split-off fragment): it follows its
        best match many-to-one, never founds a speaker, and its embedding is not absorbed. If it matches
        nobody it is left out, and its speech only keeps other speakers' reference clips clear.
        """
        if block.end <= block.start:
            return {}
        span = [(block.start, block.end)]
        exclusive = _clip(block.exclusive or _exclusive_from(block.turns), span)
        turns = _clip(block.turns or block.exclusive, span)
        fresh = self._uncovered(block.start, block.end)
        new_ex, new_all = _clip(exclusive, fresh), _clip(turns, fresh)

        labels = sorted({t.speaker for t in turns} | {t.speaker for t in exclusive} | set(block.centroids))
        cents = {lab: _unit(block.centroids.get(lab)) for lab in labels}
        talk = dict.fromkeys(labels, 0.0)
        for t in exclusive:
            talk[t.speaker] += t.end - t.start
        shared = {lab: self._shared_speech([t for t in exclusive if t.speaker == lab]) for lab in labels}
        major = [lab for lab in labels if talk[lab] >= _MIN_OWN - _EPS]
        mapping = self._link(major, cents, talk, shared)
        if len(labels) == 1 and not mapping and cents[labels[0]] is None and len(self.speakers) == 1:
            (g,) = self.speakers.values()
            if g.centroid is None:  # a diarizer that sends no embeddings: one voice throughout
                mapping[labels[0]] = g.id

        first: dict[str, float] = {}
        for t in new_all + new_ex:
            first[t.speaker] = min(first.get(t.speaker, math.inf), t.start)
        for lab in sorted((x for x in major if x in first and x not in mapping), key=lambda x: (first[x], x)):
            gid = self._new_id()
            self.speakers[gid] = GlobalSpeaker(gid, f"Speaker {gid[1:]}", None, 0.0, first[lab])
            mapping[lab] = gid
        for lab in major:
            if lab in mapping:
                self._absorb(self.speakers[mapping[lab]], cents[lab], talk[lab])
        for lab in labels:  # short labels, against every speaker including the ones just founded
            if lab not in mapping and lab not in major and (gid := self._follow(cents[lab], shared[lab])):
                mapping[lab] = gid

        for lab, gid in mapping.items():
            if lab in first:
                self.speakers[gid].first_at = min(self.speakers[gid].first_at, first[lab])
        for t in new_ex:
            if t.speaker in mapping:
                self.speakers[mapping[t.speaker]].talk_seconds += t.end - t.start
        self._ex.extend([SpeakerTurn(mapping[t.speaker], t.start, t.end) for t in new_ex if t.speaker in mapping])
        self._all.extend([SpeakerTurn(mapping[t.speaker], t.start, t.end) for t in new_all if t.speaker in mapping])
        self._stray.extend([SpeakerTurn("", t.start, t.end) for t in new_all + new_ex if t.speaker not in mapping])
        self._cover(block.start, block.end)
        return mapping

    def merge(self, src: str, into: str) -> None:
        """Fold speaker `src` into `into` (one person split in two). `src`'s id is never reused."""
        if src == into:
            return
        g = self.speakers[into]
        old = self.speakers.pop(src)
        self._absorb(g, _unit(old.centroid), self._weight.pop(src, 0.0))
        g.talk_seconds += old.talk_seconds
        g.first_at = min(g.first_at, old.first_at)
        for track in (self._ex, self._all):
            track.set([SpeakerTurn(into, t.start, t.end) if t.speaker == src else t for t in track.turns])

    def rename(self, ids: dict[str, str]) -> None:
        """Give speakers new ids (`ids`: current id -> new id; the others keep theirs), all at once, so ids may swap: to
        renumber them, or to keep the ids an earlier diarization gave the same people. A default label ("Speaker 3")
        follows its id. The new ids must not collide."""
        ids = {a: b for a, b in ids.items() if a in self.speakers and a != b}
        if not ids:
            return
        renamed = {ids.get(sid, sid): g for sid, g in self.speakers.items()}
        if len(renamed) != len(self.speakers):
            raise ValueError(f"speaker ids collide: {ids}")
        for sid, g in renamed.items():
            if g.label == f"Speaker {g.id[1:]}":
                g.label = f"Speaker {sid[1:]}"
            g.id = sid
        self.speakers = {sid: renamed[sid] for sid in sorted(renamed, key=lambda s: (_id_number(s), s))}
        self._weight = {ids.get(sid, sid): w for sid, w in self._weight.items()}
        for track in (self._ex, self._all):
            track.set([SpeakerTurn(ids.get(t.speaker, t.speaker), t.start, t.end) for t in track.turns])
        self._issued = max([self._issued, *(_id_number(s) for s in self.speakers)])

    def dump(self) -> dict:
        """The speakers' turns (with crosstalk, and exclusive), the speech of labels that matched nobody and the
        centroids, as JSON (exact floats), for `restore` (the render's diarization.json)."""
        return {"turns": [[t.speaker, t.start, t.end] for t in self._all.turns],
                "exclusive": [[t.speaker, t.start, t.end] for t in self._ex.turns],
                "stray": [[t.start, t.end] for t in self._stray.turns],
                "centroids": {sid: g.centroid.tolist() for sid, g in self.speakers.items() if g.centroid is not None}}

    @classmethod
    def restore(cls, doc: dict, start: float, end: float) -> SpeakerRegistry:
        """The registry `dump` wrote for one block over [start, end], rebuilt with one `add_block` over it and renamed
        to the ids it had: the same speakers, turns and clean speech, whatever order they were first found in."""
        turns = [SpeakerTurn(s, a, b) for s, a, b in doc["turns"]] + [SpeakerTurn("", a, b) for a, b in doc["stray"]]
        cents = {sid: np.asarray(c, dtype=np.float64) for sid, c in doc["centroids"].items()}
        reg = cls()
        mapping = reg.add_block(DiarBlock(start, end, turns, [SpeakerTurn(s, a, b) for s, a, b in doc["exclusive"]], cents))
        reg.rename({sid: label for label, sid in mapping.items()})
        return reg

    def _link(self, labels: list[str], cents: dict[str, np.ndarray | None], talk: dict[str, float],
              shared: dict[str, dict[str, float]]) -> dict[str, str]:
        """One-to-one links for the labels with enough speech to claim an id (see add_block)."""
        mapping: dict[str, str] = {}
        taken: set[str] = set()

        def take(pairs: list[tuple[float, str, str]]) -> None:
            for _, lab, gid in sorted(pairs):  # best first; ties by label, then id
                if lab not in mapping and gid not in taken:
                    mapping[lab] = gid
                    taken.add(gid)

        known = [g for g in self.speakers.values() if g.centroid is not None]
        embedded = [lab for lab in labels if cents[lab] is not None]
        if known and embedded:
            sims = np.stack([cents[lab] for lab in embedded]) @ np.stack([g.centroid for g in known]).T
            pairs = []
            for i, lab in enumerate(embedded):
                tie = _TALK_TIE * min(talk[lab], _TALK_SAT) / _TALK_SAT
                for j, g in enumerate(known):
                    s = float(sims[i, j]) + (_VOUCH_BONUS if _vouches(shared[lab], g.id) else 0.0)
                    if s >= self.link_threshold:
                        pairs.append((-(s + tie), lab, g.id))
            take(pairs)

        pairs = []
        for lab in labels:
            if lab in mapping:
                continue
            for gid, ov in shared[lab].items():
                g = self.speakers.get(gid)
                if g is None or ov < _MIN_LINK_OVERLAP or (cents[lab] is not None and g.centroid is not None):
                    continue  # when both have embeddings, the embeddings (and the overlap bonus) already said no
                pairs.append((-ov, lab, gid))
        take(pairs)
        return mapping

    def _follow(self, c: np.ndarray | None, shared: dict[str, float]) -> str | None:
        """Where a short label goes, many-to-one: its best embedding match, else (when either side has
        no embedding) the speaker it shares the most exclusive speech with, else nowhere."""
        pairs = []
        if c is not None:
            for g in self.speakers.values():
                if g.centroid is not None and (s := float(c @ g.centroid)) >= self.link_threshold:
                    pairs.append((-s, g.id))
        if not pairs:
            pairs = [(-ov, gid) for gid, ov in shared.items() if gid in self.speakers and ov >= _MIN_LINK_OVERLAP
                     and (c is None or self.speakers[gid].centroid is None)]
        return min(pairs)[1] if pairs else None

    def _shared_speech(self, turns: list[SpeakerTurn]) -> dict[str, float]:
        out: dict[str, float] = {}
        for t in turns:
            for k in self._ex.near(t.start, t.end):
                ov = min(t.end, k.end) - max(t.start, k.start)
                if ov > 0:
                    out[k.speaker] = out.get(k.speaker, 0.0) + ov
        return out

    def _absorb(self, g: GlobalSpeaker, c: np.ndarray | None, seconds: float) -> None:
        """Fold a block embedding into the speaker's centroid, weighted by the speech behind it."""
        if c is None:
            return
        w, old = max(seconds, _MIN_WEIGHT), self._weight.get(g.id, 0.0)
        mean = _unit(g.centroid * old + c * w) if g.centroid is not None and old > 0 else None
        g.centroid = mean if mean is not None else c
        self._weight[g.id] = (old if mean is not None else 0.0) + w

    def _new_id(self) -> str:
        self._issued = max([self._issued, *(_id_number(s) for s in self.speakers)]) + 1
        return f"S{self._issued}"

    # -- coverage -------------------------------------------------------------------------------

    def _cover(self, a: float, b: float) -> None:
        out: list[tuple[float, float]] = []
        for s, e in sorted([*self._coverage, (a, b)]):
            if out and s <= out[-1][1] + _JOIN:
                out[-1] = (out[-1][0], max(out[-1][1], e))
            else:
                out.append((s, e))
        self._coverage = out

    def _uncovered(self, a: float, b: float) -> list[tuple[float, float]]:
        pieces, cur = [], a
        for s, e in self._coverage:
            if e <= cur:
                continue
            if s >= b:
                break
            if s > cur:
                pieces.append((cur, s))
            cur = e
        if cur < b:
            pieces.append((cur, b))
        return [(s, e) for s, e in pieces if e - s > _EPS]

    def covered_until(self, t: float = 0.0) -> float:
        """End of the contiguous diarized coverage that contains `t` (or `t` if not covered)."""
        for s, e in self._coverage:
            if s - _JOIN <= t <= e + _JOIN:
                return max(e, t)
        return t

    def is_covered(self, a: float, b: float) -> bool:
        """Whether [a, b] is diarized. A block may end a sample short of the time it was cut at, so
        coverage within `_JOIN` of `b` counts (else the last seconds of some videos never qualify)."""
        return any(s - _JOIN <= a and b <= e + _JOIN for s, e in self._coverage)

    # -- queries --------------------------------------------------------------------------------

    def turns_in(self, a: float, b: float, exclusive: bool = True) -> list[SpeakerTurn]:
        """Turns with global ids, clipped to [a, b], in start order."""
        out = []
        for t in (self._ex if exclusive else self._all).near(a, b):
            s, e = max(t.start, a), min(t.end, b)
            if e > s:
                out.append(SpeakerTurn(t.speaker, s, e))
        return out

    def speaker_at(self, t: float, prev: str | None = None) -> str:
        """Who speaks at `t`: the exclusive turn containing it, else the nearest one within 0.5 s
        (ties go to `prev`, then to the turn about to start), else `prev`, else the most talkative."""
        best, best_key = None, None
        for k in self._ex.near(t - _NEAR, t + _NEAR):
            d = max(k.start - t, t - k.end, 0.0)
            key = (d, not (k.start <= t < k.end), k.speaker != prev, -k.start)
            if d <= _NEAR and (best_key is None or key < best_key):
                best, best_key = k.speaker, key
        if best is not None:
            return best
        if prev is not None:
            return prev
        if self.speakers:
            return max(self.speakers.values(), key=lambda g: (g.talk_seconds, -_id_number(g.id))).id
        return "S1"

    def talk_share(self) -> dict[str, float]:
        total = sum(g.talk_seconds for g in self.speakers.values())
        return {sid: (g.talk_seconds / total if total > 0 else 0.0) for sid, g in self.speakers.items()}

    def reference_clips(
        self,
        speaker: str,
        target: float = 10.0,
        min_clip: float = 2.0,
        guard: float = 0.3,
        avoid_before: float | None = 60.0,
        max_clip: float = 12.0,
    ) -> list[tuple[float, float]]:
        """Best reference spans for cloning `speaker`, best first, adding up to >= `target` s if possible.

        Each clip lies inside one of the speaker's exclusive turns (itself >= `min_clip`), trimmed by
        `guard` at both ends and kept at least `guard` from every other speaker's turn, so crosstalk
        is cut out rather than rejecting the whole turn. Clips are `min_clip`–`max_clip` long (longer
        spans are split evenly). Clips after `avoid_before` (past the cold open and intro music) come
        first; earlier ones only top up. Within each group, longer clips win (saturating at ~8 s), then
        clips further from the other speakers.
        """
        if target <= 0:
            return []
        late: list[tuple[float, float, float]] = []
        early: list[tuple[float, float, float]] = []
        for t in self._ex.turns:
            if t.speaker != speaker or t.end - t.start < min_clip:
                continue
            spans = [(t.start + guard, t.end - guard)]
            for o in self._others(speaker, t.start - guard, t.end + guard):
                spans = _subtract(spans, o.start - guard, o.end + guard)
            for a, b in spans:
                if avoid_before is None:
                    parts = [(a, b, late)]
                else:
                    parts = [(a, min(b, avoid_before), early), (max(a, avoid_before), b, late)]
                for pa, pb, pool in parts:
                    for ca, cb in _split(pa, pb, max_clip):
                        if cb - ca >= min_clip - _EPS:
                            pool.append((-self._clip_score(speaker, ca, cb), ca, cb))
        out, total = [], 0.0
        for _, a, b in sorted(late) + sorted(early):
            if total >= target - _EPS:
                break
            out.append((a, b))
            total += b - a
        return out

    def best_span(
        self,
        speaker: str,
        length: float = 10.0,
        min_len: float = 6.0,
        guard: float = 0.3,
        avoid_before: float | None = None,
        score: Callable[[float, float], float] | None = None,
    ) -> tuple[float, float] | None:
        """The best single contiguous span of `speaker`'s speech for the timbre reference, or None.

        One unbroken slice of audio, never stitched: the speaker's exclusive turns, joined across pauses
        of up to 0.6 s (diarized, so no seek hole), cut at least `guard` clear of every other speaker's
        turn (crosstalk and unmatched short labels included). Where silence borders it, it is not trimmed,
        so it does not start or end inside the speaker's first or last word. A clean stretch longer than
        `length` offers the windows of `min_len`–`length` s whose edges fall at its ends or in its pauses
        (the middle of a gap between two of the speaker's turns), so no edge cuts a word; only a stretch
        with no such window (speech the diarizer shows no pause in) offers windows of exactly `length`,
        spread evenly over it. A stretch shorter than `min_len` offers nothing.

        Each candidate scores up to 1 for its seconds of the speaker's speech (saturating at `length`)
        plus up to 0.3 for isolation (2 s clear of other speakers). With `score`, the caller's score(a, b)
        is added for a batch of 8 candidates, then the next batch only if every one of those was rejected
        (a NaN, infinite or None score rejects a span, e.g. music under it): so a score on a 0–1 scale (an
        SNR estimate, cosine to the speaker's centroid) weighs about as much as length. A batch takes the
        best candidates; where more than 8 are within 0.05 of each other (near-ties, as in a long
        monologue), it takes 8 of those spread evenly over the timeline, so the caller's score, not the
        earliest start, picks among them. With `avoid_before`, spans after it (past a cold open and intro
        music) win; earlier ones are used only when there is nothing later.
        """
        if length <= 0:
            return None
        min_len = min(min_len, length)
        late: list[tuple[float, float, float]] = []
        early: list[tuple[float, float, float]] = []
        for a, b, pauses in self._stretches(speaker, guard):
            parts = [(a, b, late)] if avoid_before is None else [(a, min(b, avoid_before), early),
                                                                  (max(a, avoid_before), b, late)]
            for pa, pb, pool in parts:
                if pb - pa >= min_len - _EPS:
                    for wa, wb in _windows(pa, pb, length, min_len, pauses):
                        pool.append((round(self._span_score(speaker, wa, wb, length), 6), wa, wb))
        for pool in (late, early):
            if score is None:
                if pool:
                    _, a, b = min(pool, key=lambda c: (-c[0], c[1]))
                    return a, b
                continue
            for batch in _batches(pool, _SCORE_TOP, _SCORE_TIE):
                best = None
                for base, a, b in batch:
                    extra = score(a, b)
                    if extra is None or not math.isfinite(extra):
                        continue
                    total = round(base + float(extra), 6)
                    if best is None or total > best[0]:
                        best = (total, a, b)
                if best is not None:
                    return best[1], best[2]
        return None

    def clean_clips(
        self,
        speaker: str,
        max_total: float = 60.0,
        min_clip: float = 2.0,
        guard: float = 0.3,
        avoid_before: float | None = None,
    ) -> list[tuple[float, float]]:
        """Clean clips of `speaker`, best first, adding up to at most `max_total` s: for averaging a speaker
        embedding over much of their speech (a track clone), where no single clip needs to be long.

        Clips come from the same clean stretches as best_span (contiguous speech, `guard` clear of other
        speakers, untrimmed at silence), cut into pieces of up to 10 s; pieces under `min_clip` are
        dropped. They rank by seconds of speech (saturating at 8 s), then isolation, then time. The last
        clip is trimmed about its middle to fit `max_total` when at least `min_clip` of room is left. With
        `avoid_before`, clips after it come first and earlier ones only top up.
        """
        late: list[tuple[float, float, float]] = []
        early: list[tuple[float, float, float]] = []
        for a, b, _ in self._stretches(speaker, guard):
            parts = [(a, b, late)] if avoid_before is None else [(a, min(b, avoid_before), early),
                                                                  (max(a, avoid_before), b, late)]
            for pa, pb, pool in parts:
                for ca, cb in _split(pa, pb, _CLEAN_MAX):
                    if cb - ca >= min_clip - _EPS:
                        pool.append((-round(self._span_score(speaker, ca, cb, _LENGTH_SAT), 6), ca, cb))
        out: list[tuple[float, float]] = []
        total = 0.0
        for _, a, b in sorted(late) + sorted(early):
            room = max_total - total
            if room < min_clip - _EPS:
                break
            if b - a > room:
                mid = (a + b) / 2
                a, b = mid - room / 2, mid + room / 2
            out.append((a, b))
            total += b - a
        return out

    def _stretches(self, speaker: str, guard: float) -> list[tuple[float, float, list[float]]]:
        """`speaker`'s clean contiguous stretches, each with the pauses inside it (the middle of each bridged
        gap): exclusive turns joined across diarized pauses of up to _BRIDGE s and cut `guard` clear of
        every other speaker's turn. Where silence borders a stretch it is not trimmed, so it keeps the
        speaker's first and last word whole."""
        runs: list[tuple[float, float, list[float]]] = []
        for t in self._ex.turns:
            if t.speaker != speaker:
                continue
            if runs and t.start - runs[-1][1] <= _BRIDGE and self.is_covered(runs[-1][1], t.start):
                a, b, pauses = runs[-1]
                if t.start > b:
                    pauses.append((b + t.start) / 2)
                runs[-1] = (a, max(b, t.end), pauses)
            else:
                runs.append((t.start, t.end, []))
        out: list[tuple[float, float, list[float]]] = []
        for a, b, pauses in runs:
            spans = [(a, b)]
            for o in self._others(speaker, a - guard, b + guard):
                spans = _subtract(spans, o.start - guard, o.end + guard)
            out.extend((s, e, [p for p in pauses if s < p < e]) for s, e in spans if e - s > _EPS)
        return out

    def _others(self, speaker: str, a: float, b: float) -> list[SpeakerTurn]:
        return [t for track in (self._all, self._ex, self._stray) for t in track.near(a, b) if t.speaker != speaker]

    def _isolation(self, speaker: str, a: float, b: float) -> float:
        """Seconds from [a, b] to the nearest other speaker's turn, capped at _ISOLATION_CAP."""
        iso = _ISOLATION_CAP
        for o in self._others(speaker, a - _ISOLATION_CAP, b + _ISOLATION_CAP):
            iso = min(iso, max(o.start - b, a - o.end, 0.0))
        return iso

    def _clip_score(self, speaker: str, a: float, b: float) -> float:
        iso = self._isolation(speaker, a, b)
        return min(b - a, _LENGTH_SAT) / _LENGTH_SAT + _ISOLATION_WEIGHT * iso / _ISOLATION_CAP

    def _span_score(self, speaker: str, a: float, b: float, sat: float) -> float:
        """Like _clip_score, but counting the speaker's speech inside [a, b] (saturating at `sat`), so a
        span bridged across pauses is not rated by its silence."""
        voiced = sum(max(0.0, min(t.end, b) - max(t.start, a)) for t in self._ex.near(a, b) if t.speaker == speaker)
        return min(voiced, sat) / sat + _ISOLATION_WEIGHT * self._isolation(speaker, a, b) / _ISOLATION_CAP


# -- the whole-file render (OFFLINE-RENDER §2.3) --------------------------------------------------------------------------

def settle(registry: SpeakerRegistry, *, min_talk: float = MIN_TALK, min_share: float = MIN_SHARE,
           same_cos: float = SAME_COS, floor_cap: float = FLOOR_CAP, hinted: bool = False) -> list[tuple[str, str, str]]:
    """Settle a whole-file diarization (registered with one `add_block`) into the speakers the dub voices, in place.

    Unless the user gave the count (`hinted`), fold over-split speakers back, with `SpeakerRegistry.merge`:
    - a speaker with less exclusive talk than max(`min_talk`, `min_share` x everyone's), but never more than
      `floor_cap` x everyone's (a short clip), goes into the speaker nearest by centroid cosine, or with no centroid on
      either side, into the one whose speech lies most next to theirs (the least talkative first; `talk`);
    - then the most similar pair with centroid cosine >= `same_cos`, the one who talks less into the other, until no
      pair is left (`same`).
    With a count nothing merges: the user's number wins. Then the ids are renumbered S1..Sk in order of first speech.
    Returns the merges as (from, into, why), in the ids they had before renumbering."""
    merged: list[tuple[str, str, str]] = []
    if not hinted:
        total = sum(g.talk_seconds for g in registry.speakers.values())
        floor = min(max(min_talk, min_share * total), floor_cap * total)
        while len(registry.speakers) > 1:
            small = min((g for g in registry.speakers.values() if g.talk_seconds < floor),
                        key=lambda g: (g.talk_seconds, _id_number(g.id)), default=None)
            if small is None:
                break
            into = _nearest(registry, small.id)
            registry.merge(small.id, into)
            merged.append((small.id, into, "talk"))
        while len(registry.speakers) > 1:
            known = sorted((g for g in registry.speakers.values() if g.centroid is not None), key=lambda g: _id_number(g.id))
            pairs = [(float(a.centroid @ b.centroid), a, b) for i, a in enumerate(known) for b in known[i + 1:]]
            best = max((p for p in pairs if p[0] >= same_cos), key=lambda p: p[0], default=None)
            if best is None:
                break
            src, into = sorted(best[1:], key=lambda g: (g.talk_seconds, -_id_number(g.id)))
            registry.merge(src.id, into.id)
            merged.append((src.id, into.id, "same"))
    order = sorted(registry.speakers.values(), key=lambda g: (g.first_at, _id_number(g.id)))
    registry.rename({g.id: f"S{k}" for k, g in enumerate(order, 1)})
    return merged


def _nearest(registry: SpeakerRegistry, sid: str) -> str:
    """The speaker `sid` merges into: the nearest by centroid cosine; with no centroid on either side, the one with the
    most exclusive speech within _ADJACENT s of theirs; else the most talkative."""
    g = registry.speakers[sid]
    others = sorted((o for o in registry.speakers.values() if o.id != sid), key=lambda o: _id_number(o.id))
    if g.centroid is not None:
        scored = [(float(g.centroid @ o.centroid), o.id) for o in others if o.centroid is not None]
        if scored:
            return max(scored, key=lambda x: x[0])[1]
    near: dict[str, float] = {}
    for t in registry._ex.turns:
        if t.speaker == sid:
            for o in registry.turns_in(t.start - _ADJACENT, t.end + _ADJACENT):
                if o.speaker != sid:
                    near[o.speaker] = near.get(o.speaker, 0.0) + o.end - o.start
    if near:
        return max(others, key=lambda o: near.get(o.id, 0.0)).id
    return max(others, key=lambda o: o.talk_seconds).id


def match_ids(previous_exclusive: list[tuple[str, float, float]], registry: SpeakerRegistry) -> dict[str, str]:
    """The ids to give a re-run's speakers (current id -> id), so the same people keep their ids (OFFLINE-RENDER §2.3):
    one to one, the assignment that maximises the exclusive speech each speaker shares with the id's previous turns
    (`previous_exclusive`: (id, start, end)), by brute force (k <= 6). Only pairs that share speech are matched; ties
    go to the earlier speaker and the lower id. A speaker left unmatched (a split, a new voice) takes the next id no
    earlier speaker had."""
    shared: dict[tuple[str, str], float] = {}
    for old, a, b in previous_exclusive:
        for t in registry.turns_in(a, b):
            shared[t.speaker, old] = shared.get((t.speaker, old), 0.0) + t.end - t.start
    new = sorted(registry.speakers.values(), key=lambda g: (g.first_at, _id_number(g.id)))
    olds = sorted({old for old, _, _ in previous_exclusive}, key=_id_number)
    best: list = [-1.0, ()]

    def assign(i: int, used: frozenset, total: float, picks: tuple) -> None:
        if i == len(new):
            if total > best[0] + _EPS:
                best[:] = [total, picks]
            return
        for old in olds:
            if old not in used and (s := shared.get((new[i].id, old), 0.0)) > 0:
                assign(i + 1, used | {old}, total + s, picks + (old,))
        assign(i + 1, used, total, picks + (None,))

    assign(0, frozenset(), 0.0, ())
    ids = {g.id: old for g, old in zip(new, best[1]) if old is not None}
    n = max([0, *(_id_number(s) for s in olds)])
    for g in new:
        if g.id not in ids:
            n += 1
            ids[g.id] = f"S{n}"
    return ids


def activity(registry: SpeakerRegistry, sid: str, duration: float, bins: int = 120) -> list[float]:
    """`sid`'s share of each of `bins` equal slices of [0, `duration`]: seconds of their exclusive speech in the slice
    over its length (the UI's activity strip)."""
    out = [0.0] * bins
    if duration <= 0:
        return out
    width = duration / bins
    for t in registry.turns_in(0.0, duration):
        if t.speaker != sid:
            continue
        for k in range(min(int(t.start // width), bins - 1), min(int(t.end // width), bins - 1) + 1):
            out[k] += max(0.0, min(t.end, (k + 1) * width) - max(t.start, k * width))
    return [x / width for x in out]


def _unit(v: object) -> np.ndarray | None:
    """`v` as a unit float64 vector, or None when it is missing, empty, NaN/inf or zero."""
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float64).ravel()
    if a.size == 0 or not np.all(np.isfinite(a)):
        return None
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-8 else None


def _vouches(shared: dict[str, float], gid: str) -> bool:
    """The label shares most of its overlap speech, and at least 2 s of it, with speaker `gid`."""
    ov = shared.get(gid, 0.0)
    return ov >= _VOUCH and ov > sum(shared.values()) / 2


def _id_number(sid: str) -> int:
    return int(sid[1:]) if sid[:1] == "S" and sid[1:].isdigit() else 0


def _clip(turns: list[SpeakerTurn], pieces: list[tuple[float, float]]) -> list[SpeakerTurn]:
    out = []
    for t in turns:
        for a, b in pieces:
            s, e = max(t.start, a), min(t.end, b)
            if e - s >= _MIN_PIECE:
                out.append(SpeakerTurn(t.speaker, float(s), float(e)))
    return out


def _exclusive_from(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Non-overlapping turns from overlapping ones: where speakers overlap, the longer turn wins."""
    cuts = sorted({x for t in turns for x in (t.start, t.end)})
    out: list[SpeakerTurn] = []
    for a, b in zip(cuts, cuts[1:]):
        mid = (a + b) / 2
        live = [t for t in turns if t.start <= mid < t.end]
        if not live:
            continue
        spk = max(live, key=lambda t: (t.end - t.start, -t.start)).speaker
        if out and out[-1].speaker == spk and out[-1].end == a:
            out[-1] = SpeakerTurn(spk, out[-1].start, b)
        else:
            out.append(SpeakerTurn(spk, a, b))
    return out


def _subtract(spans: list[tuple[float, float]], a: float, b: float) -> list[tuple[float, float]]:
    out = []
    for s, e in spans:
        if b <= s or a >= e:
            out.append((s, e))
            continue
        if a > s:
            out.append((s, a))
        if b < e:
            out.append((b, e))
    return out


def _windows(a: float, b: float, size: float, min_size: float = 0.0,
             pauses: list[float] | None = None) -> list[tuple[float, float]]:
    """Candidate windows in [a, b]: [a, b] itself when it is at most `size` long. Else, from each of a and
    the `pauses` inside (points in silence), the longest window of `min_size`–`size` s that ends at a later
    pause or at b, so neither edge falls inside a word. With no such window, windows of exactly `size`
    spread evenly over [a, b], the first starting at a and the last ending at b (enough of them to cover it)."""
    if b - a <= size + _EPS:
        return [(a, b)]
    edges = [a, *sorted(p for p in pauses or () if a < p < b), b]
    snapped = []
    for i, s in enumerate(edges[:-1]):
        e = max((e for e in edges[i + 1:] if e - s <= size + _EPS), default=s)
        if e - s >= min_size - _EPS and e > s:
            snapped.append((s, e))
    if snapped:
        return snapped
    n = max(2, math.ceil((b - a) / size - _EPS))
    step = (b - a - size) / (n - 1)
    return [(a + i * step, a + i * step + size) if i < n - 1 else (b - size, b) for i in range(n)]


def _batches(pool: list[tuple[float, float, float]], size: int, tie: float) -> Iterator[list[tuple[float, float, float]]]:
    """Candidates (score, a, b) in batches of `size`, best first. Each batch takes the best remaining
    candidates; when more than `size` of them score within `tie` of the best one, it takes `size` of those
    spread evenly over time instead of the earliest."""
    rest = sorted(pool, key=lambda c: (-c[0], c[1]))
    while rest:
        tier = sorted((c for c in rest if c[0] >= rest[0][0] - tie), key=lambda c: c[1])
        if len(tier) > size:
            batch = [tier[int((i + 0.5) * len(tier) / size)] for i in range(size)]
        else:
            batch = rest[:size]
        yield batch
        taken = set(map(id, batch))
        rest = [c for c in rest if id(c) not in taken]


def _split(a: float, b: float, size: float) -> list[tuple[float, float]]:
    """[a, b] cut into equal pieces no longer than `size`."""
    if b <= a:
        return []
    n = max(1, math.ceil((b - a) / size - _EPS))
    step = (b - a) / n
    return [(a + i * step, b if i == n - 1 else a + (i + 1) * step) for i in range(n)]
