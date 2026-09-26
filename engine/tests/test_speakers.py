import math

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from maata_engine.speakers import DiarBlock, GlobalSpeaker, SpeakerRegistry
from maata_engine.types import SpeakerTurn

DIM = 256


def voices(n, seed=0):
    """n orthonormal 256-d embeddings: n different people (cosine similarity 0)."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((DIM, n)))
    return [q[:, i].copy() for i in range(n)]


def near(v, sim, seed=1):
    """A unit vector with cosine similarity exactly `sim` to the unit vector `v`."""
    u = np.random.default_rng(seed).standard_normal(DIM)
    u -= (u @ v) * v
    u /= np.linalg.norm(u)
    return sim * v + math.sqrt(1 - sim * sim) * u


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def T(spk, s, e):
    return SpeakerTurn(spk, s, e)


def block(start, end, exclusive, centroids, turns=None):
    return DiarBlock(start, end, list(exclusive if turns is None else turns), list(exclusive), dict(centroids))


def spans(turns):
    return [(t.speaker, round(t.start, 6), round(t.end, 6)) for t in turns]


A, B, C, D4, E5 = voices(5)


# -- linking --------------------------------------------------------------------------------------

def test_first_block_creates_speakers_in_order_of_first_speech():
    reg = SpeakerRegistry()
    m = reg.add_block(block(0, 60, [T("SPEAKER_01", 1, 20), T("SPEAKER_00", 21, 40), T("SPEAKER_01", 41, 59)],
                            {"SPEAKER_00": B, "SPEAKER_01": A}))
    assert m == {"SPEAKER_01": "S1", "SPEAKER_00": "S2"}
    s1, s2 = reg.speakers["S1"], reg.speakers["S2"]
    assert (s1.label, s2.label) == ("Speaker 1", "Speaker 2")
    assert (s1.first_at, s2.first_at) == (1, 21)
    assert s1.talk_seconds == pytest.approx(37) and s2.talk_seconds == pytest.approx(19)
    assert cos(s1.centroid, A) == pytest.approx(1) and np.linalg.norm(s1.centroid) == pytest.approx(1)
    assert reg.talk_share() == pytest.approx({"S1": 37 / 56, "S2": 19 / 56})
    assert reg.covered_until() == 60


def test_overlapping_block_relinks_swapped_labels_without_doubling_turns():
    reg = SpeakerRegistry()
    # Block 1 ends mid-turn: A really talks 71–110.
    reg.add_block(block(0, 100, [T("SPEAKER_00", 0, 40), T("SPEAKER_01", 41, 70), T("SPEAKER_00", 71, 100)],
                        {"SPEAKER_00": A, "SPEAKER_01": B}))
    # Block 2 starts 15 s earlier than block 1 ends; pyannote numbered the voices the other way round.
    m = reg.add_block(block(85, 185, [T("SPEAKER_01", 85, 110), T("SPEAKER_00", 111, 150), T("SPEAKER_01", 151, 185)],
                            {"SPEAKER_00": near(B, 0.75), "SPEAKER_01": near(A, 0.8)}))
    assert m == {"SPEAKER_01": "S1", "SPEAKER_00": "S2"}
    assert set(reg.speakers) == {"S1", "S2"}
    assert spans(reg.turns_in(0, 185)) == spans(
        [T("S1", 0, 40), T("S2", 41, 70), T("S1", 71, 110), T("S2", 111, 150), T("S1", 151, 185)])
    assert spans(reg.turns_in(0, 185, exclusive=False)) == spans(reg.turns_in(0, 185))
    assert reg.speakers["S1"].talk_seconds == pytest.approx(40 + 39 + 34)  # 85–100 counted once
    assert reg.speakers["S2"].talk_seconds == pytest.approx(29 + 39)
    assert reg.covered_until(0) == reg.covered_until(90) == 185


def test_speaker_silent_for_several_blocks_returns_with_the_same_id():
    reg = SpeakerRegistry()
    reg.add_block(block(0, 100, [T("SPEAKER_00", 0, 50), T("SPEAKER_01", 51, 100)], {"SPEAKER_00": B, "SPEAKER_01": A}))
    # B is silent for two blocks; A is SPEAKER_00 there (the label B had in block 1).
    for k, (a, b) in enumerate([(90, 190), (180, 280)]):
        m = reg.add_block(block(a, b, [T("SPEAKER_00", a, b)], {"SPEAKER_00": near(A, 0.7, seed=10 + k)}))
        assert m == {"SPEAKER_00": "S2"}
    m = reg.add_block(block(270, 370, [T("SPEAKER_01", 270, 300), T("SPEAKER_00", 301, 370)],
                            {"SPEAKER_00": near(B, 0.65, seed=20), "SPEAKER_01": near(A, 0.7, seed=21)}))
    assert m == {"SPEAKER_00": "S1", "SPEAKER_01": "S2"}
    assert set(reg.speakers) == {"S1", "S2"}
    assert reg.speaker_at(330) == "S1" and reg.speaker_at(150) == "S2"


def test_new_third_speaker_gets_the_next_id():
    reg = SpeakerRegistry()
    reg.add_block(block(0, 100, [T("x", 0, 50), T("y", 51, 100)], {"x": A, "y": B}))
    m = reg.add_block(block(100, 200, [T("p", 100, 130), T("q", 131, 160), T("r", 161, 200)],
                            {"p": near(B, 0.8), "q": C, "r": near(A, 0.8)}))
    assert m == {"p": "S2", "q": "S3", "r": "S1"}
    g = reg.speakers["S3"]
    assert (g.label, g.first_at, g.talk_seconds) == ("Speaker 3", 131, pytest.approx(29))


@pytest.mark.parametrize("sim_x, sim_y, x_gets", [(0.9, 0.7, "S1"), (0.7, 0.9, "S3")])
def test_assignment_is_one_to_one_best_pair_first(sim_x, sim_y, x_gets):
    reg = SpeakerRegistry()
    reg.add_block(block(0, 100, [T("a", 0, 50), T("b", 51, 100)], {"a": A, "b": B}))
    # Both labels resemble A; only the closer one may take S1, the other must not merge into it.
    m = reg.add_block(block(100, 200, [T("x", 100, 150), T("y", 151, 200)],
                            {"x": near(A, sim_x, seed=3), "y": near(A, sim_y, seed=4)}))
    y_gets = "S3" if x_gets == "S1" else "S1"
    assert m == {"x": x_gets, "y": y_gets}
    assert set(reg.speakers) == {"S1", "S2", "S3"}


def test_link_threshold():
    first = block(0, 100, [T("a", 0, 100)], {"a": A})
    later = block(100, 200, [T("b", 100, 200)], {"b": near(A, 0.45)})
    reg = SpeakerRegistry()
    reg.add_block(first)
    assert reg.add_block(later) == {"b": "S2"}
    loose = SpeakerRegistry(link_threshold=0.4)
    loose.add_block(first)
    assert loose.add_block(later) == {"b": "S1"}


def test_centroid_is_a_talk_weighted_running_mean():
    reg = SpeakerRegistry()
    reg.add_block(block(0, 100, [T("a", 0, 60), T("b", 60, 100)], {"a": A, "b": B}))
    v = near(A, 0.8)
    reg.add_block(block(100, 200, [T("a", 100, 130), T("b", 130, 200)], {"a": v, "b": B}))
    want = 60 * A + 30 * v
    np.testing.assert_allclose(reg.speakers["S1"].centroid, want / np.linalg.norm(want), atol=1e-9)
    np.testing.assert_allclose(reg.speakers["S2"].centroid, B, atol=1e-9)


@pytest.mark.parametrize("frag_len, frag_gets", [(0.6, "S2"), (3.0, "S3")])
def test_split_off_label_with_a_closer_embedding_cannot_take_the_main_speakers_id(frag_len, frag_gets):
    reg = SpeakerRegistry()
    reg.add_block(block(0, 180, [T("x", 0, 90), T("y", 90, 180)], {"x": A, "y": B}))
    # pyannote split B in two: "m" holds almost all of B's speech, the fragment's embedding is a hair closer to B.
    # A 0.6 s fragment just follows S2; a 3 s one is kept apart (someone new) but must not push "m" out of S2.
    f = 300 + frag_len
    m = reg.add_block(block(180, 360, [T("m", 180, 300), T("frag", 300, f), T("n", f, 360)],
                            {"m": near(B, 0.70, seed=2), "frag": near(B, 0.72, seed=3), "n": near(A, 0.8)}))
    assert m == {"m": "S2", "frag": frag_gets, "n": "S1"}
    assert reg.speakers["S2"].talk_seconds == pytest.approx(90 + 120 + (frag_len if frag_gets == "S2" else 0))


@pytest.mark.parametrize("sim, y_from, p_gets", [
    (0.48, 90, "S2"),    # drifted just below the threshold: the 15 s it shares with S2 vouch for it
    (0.25, 90, "S3"),    # the embedding clearly disagrees: shared speech alone doesn't link it
    (0.48, 175, "S3"),   # only 5 of its 15 s of overlap speech were S2's: no vouching
])
def test_shared_overlap_speech_backs_up_an_embedding_that_drifted_below_the_threshold(sim, y_from, p_gets):
    reg = SpeakerRegistry()
    reg.add_block(block(0, 180, [T("x", 0, y_from), T("y", y_from, 180)], {"x": A, "y": B}))
    m = reg.add_block(block(165, 360, [T("p", 165, 250), T("q", 250, 360)], {"p": near(B, sim), "q": near(A, 0.8)}))
    assert m == {"p": p_gets, "q": "S1"}


# -- ids ------------------------------------------------------------------------------------------

def test_new_ids_never_reuse_or_collide_after_a_speaker_is_removed():
    reg = SpeakerRegistry()
    reg.add_block(block(0, 60, [T("a", 0, 10), T("b", 10, 20), T("c", 20, 30)], {"a": A, "b": B, "c": C}))
    assert sorted(reg.speakers) == ["S1", "S2", "S3"]
    del reg.speakers["S2"]
    # A length-based id (len + 1 = "S3") would overwrite the live S3; reusing "S2" would revive a dead id.
    m = reg.add_block(block(60, 120, [T("d", 60, 120)], {"d": D4}))
    assert m == {"d": "S4"}
    assert sorted(reg.speakers) == ["S1", "S3", "S4"]
    assert reg.speakers["S3"].first_at == 20


def test_merge_folds_speakers_and_retires_the_id():
    reg = SpeakerRegistry()
    reg.add_block(block(0, 60, [T("a", 0, 20), T("b", 20, 30), T("a", 31, 60)], {"a": A, "b": B}))
    reg.merge("S2", "S1")
    assert list(reg.speakers) == ["S1"]
    assert spans(reg.turns_in(0, 60)) == spans([T("S1", 0, 30), T("S1", 31, 60)])  # touching pieces joined
    assert reg.speakers["S1"].talk_seconds == pytest.approx(59)
    want = 49 * A + 10 * B
    np.testing.assert_allclose(reg.speakers["S1"].centroid, want / np.linalg.norm(want), atol=1e-9)
    assert reg.add_block(block(60, 120, [T("e", 60, 120)], {"e": E5})) == {"e": "S3"}


def test_new_id_follows_the_highest_existing_number():
    reg = SpeakerRegistry()
    reg.add_block(block(0, 60, [T("a", 0, 60)], {"a": A}))
    reg.speakers["S9"] = GlobalSpeaker("S9", "Guest", None, 0.0, 0.0)
    assert reg.add_block(block(60, 120, [T("b", 60, 120)], {"b": B})) == {"b": "S10"}


# -- missing centroids ----------------------------------------------------------------------------

def test_labels_without_a_usable_centroid_link_by_shared_speech():
    reg = SpeakerRegistry()
    reg.add_block(block(0, 100, [T("a", 0, 50), T("b", 50, 100)], {"a": A, "b": B}))
    nan = np.full(DIM, np.nan)
    m = reg.add_block(block(85, 185, [T("x", 85, 120), T("y", 121, 160), T("z", 161, 185)],
                            {"x": nan, "y": near(A, 0.8), "z": np.zeros(DIM)}))
    # x shares 85–100 with S2; z has no embedding and no shared speech, so it is someone new.
    assert m == {"x": "S2", "y": "S1", "z": "S3"}
    np.testing.assert_allclose(reg.speakers["S2"].centroid, B, atol=1e-9)  # NaN never absorbed
    assert reg.speakers["S3"].centroid is None
    assert reg.speakers["S2"].talk_seconds == pytest.approx(50 + 20)


def test_label_missing_from_centroids_links_by_overlap_one_to_one():
    reg = SpeakerRegistry()
    reg.add_block(block(0, 100, [T("a", 80, 90), T("b", 90, 100)], {"a": A, "b": B}))
    # Neither label has an embedding; each takes the known speaker it shares the most speech with.
    m = reg.add_block(block(80, 180, [T("p", 80, 91), T("q", 91, 150), T("p", 150, 180)], {}))
    assert m == {"p": "S1", "q": "S2"}


def test_unlinked_label_with_no_new_speech_is_not_registered():
    reg = SpeakerRegistry()
    reg.add_block(block(0, 100, [T("a", 0, 50)], {"a": A}))
    # "q" only speaks inside the overlap, where block 1 heard nobody: nothing new to register.
    m = reg.add_block(block(85, 185, [T("q", 88, 99), T("r", 110, 180)], {"r": near(A, 0.8)}))
    assert m == {"r": "S1"}
    assert set(reg.speakers) == {"S1"}


def test_embedded_label_links_by_overlap_only_to_a_speaker_without_embedding():
    reg = SpeakerRegistry()
    reg.add_block(block(0, 100, [T("a", 0, 100)], {}))       # S1 has no embedding yet
    assert reg.add_block(block(90, 190, [T("b", 90, 190)], {"b": A})) == {"b": "S1"}
    np.testing.assert_allclose(reg.speakers["S1"].centroid, A, atol=1e-9)

    reg = SpeakerRegistry()
    reg.add_block(block(0, 100, [T("a", 0, 100)], {"a": B}))
    assert reg.add_block(block(90, 190, [T("b", 90, 190)], {"b": A})) == {"b": "S2"}  # voices disagree


def test_blocks_without_embeddings_keep_one_voice_across_seek_gaps():
    # The single-speaker fallback sends one label and no embedding per block; after a seek there is no overlap either.
    reg = SpeakerRegistry()
    for a, b in [(0, 180), (165, 360), (1985, 2180), (985, 1180)]:
        assert reg.add_block(block(a, b, [T("S1", a, b)], {})) == {"S1": "S1"}
    assert list(reg.speakers) == ["S1"]
    assert reg.speakers["S1"].talk_seconds == pytest.approx(360 + 195 + 195)
    # With embeddings in play, a lone label with no evidence either way is still someone new.
    reg = SpeakerRegistry()
    reg.add_block(block(0, 180, [T("a", 0, 180)], {"a": A}))
    assert reg.add_block(block(1985, 2180, [T("b", 1985, 2180)], {})) == {"b": "S2"}


# -- short labels ---------------------------------------------------------------------------------

def test_short_labels_follow_their_best_match_many_to_one():
    reg = SpeakerRegistry()
    # "b2" (0.5 s) was split off B in the very first block: it follows S2, founded in the same block.
    m = reg.add_block(block(0, 100, [T("a", 0, 45), T("b2", 45, 45.5), T("b", 45.5, 100)],
                            {"a": A, "b": B, "b2": near(B, 0.9)}))
    assert m == {"a": "S1", "b": "S2", "b2": "S2"}
    np.testing.assert_allclose(reg.speakers["S2"].centroid, B, atol=1e-9)  # a fragment's embedding isn't absorbed
    assert reg.speakers["S2"].talk_seconds == pytest.approx(55)
    assert spans(reg.turns_in(0, 100)) == spans([T("S1", 0, 45), T("S2", 45, 100)])
    # "x" (1 s, no embedding) shares its speech with S2, which "y" has already claimed.
    m = reg.add_block(block(85, 185, [T("x", 85, 86), T("y", 86, 150), T("z", 150, 185)],
                            {"y": near(B, 0.8), "z": near(A, 0.8)}))
    assert m == {"x": "S2", "y": "S2", "z": "S1"}
    assert set(reg.speakers) == {"S1", "S2"}


def test_short_label_that_matches_nobody_is_left_out_but_kept_out_of_reference_clips():
    reg = SpeakerRegistry()
    excl = [T("a", 0, 30), T("b", 31, 110)]
    # "c" laughs over B for 0.8 s and never speaks otherwise: no third speaker for a two-person podcast.
    m = reg.add_block(block(0, 120, excl, {"a": A, "b": B, "c": C}, turns=excl + [T("c", 70, 70.8)]))
    assert m == {"a": "S1", "b": "S2"}
    assert set(reg.speakers) == {"S1", "S2"}
    assert spans(reg.turns_in(0, 120, exclusive=False)) == spans([T("S1", 0, 30), T("S2", 31, 110)])
    clips = reg.reference_clips("S2", target=80)
    assert clips and all(b + 0.3 <= 70 + 1e-9 or 70.8 + 0.3 <= a + 1e-9 for a, b in clips)


# -- coverage and turns ---------------------------------------------------------------------------

def test_coverage_with_gaps_and_out_of_order_blocks():
    reg = SpeakerRegistry()
    reg.add_block(block(0, 100, [T("a", 0, 100)], {"a": A}))
    reg.add_block(block(90, 200, [T("a", 90, 200)], {"a": A}))
    reg.add_block(block(300, 400, [T("a", 300, 400)], {"a": A}))
    assert reg.covered_until() == 200
    assert reg.covered_until(250) == 250 and reg.covered_until(350) == 400
    assert reg.is_covered(10, 150) and not reg.is_covered(150, 250) and not reg.is_covered(250, 260)
    # A later block fills the gap; its turn is clipped at both sides and joins its neighbours.
    reg.add_block(block(195, 305, [T("a", 190, 310)], {"a": A}))
    assert reg.covered_until() == 400 and reg.is_covered(0, 400)
    assert spans(reg.turns_in(0, 400)) == spans([T("S1", 0, 400)])
    assert reg.speakers["S1"].talk_seconds == pytest.approx(400)


def test_coverage_allows_for_block_ends_on_whole_samples():
    # The session cuts blocks at whole 16 kHz samples, so the last one can end a sample short of the video.
    dur = 4_096_003 / 16_000                                   # 256.0001875 s
    reg = SpeakerRegistry()
    reg.add_block(block(0, 180, [T("a", 0, 180)], {"a": A}))
    reg.add_block(block(165, 256.000125, [T("a", 165, 256.000125)], {"a": A}))
    assert reg.is_covered(0, dur) and reg.is_covered(240, dur)
    assert reg.covered_until() == pytest.approx(dur, abs=1e-4) and reg.covered_until(dur) == dur
    assert not reg.is_covered(0, dur + 0.01)


def test_turns_in_clips_and_can_include_crosstalk():
    reg = SpeakerRegistry()
    reg.add_block(block(0, 20, [T("a", 0, 9), T("b", 9, 12)], {"a": A, "b": B},
                        turns=[T("a", 0, 10), T("b", 8, 12)]))
    assert spans(reg.turns_in(5, 11)) == spans([T("S1", 5, 9), T("S2", 9, 11)])
    assert spans(reg.turns_in(5, 11, exclusive=False)) == spans([T("S1", 5, 10), T("S2", 8, 11)])
    assert reg.turns_in(13, 20) == []


def test_exclusive_turns_are_derived_when_the_block_has_none():
    reg = SpeakerRegistry()
    blk = DiarBlock(0, 20, [T("a", 0, 10), T("b", 8, 12)], [], {"a": A, "b": B})
    reg.add_block(blk)
    assert spans(reg.turns_in(0, 20)) == spans([T("S1", 0, 10), T("S2", 10, 12)])  # longer turn wins the overlap
    assert reg.talk_share() == pytest.approx({"S1": 10 / 12, "S2": 2 / 12})


# -- speaker_at -----------------------------------------------------------------------------------

def test_speaker_at_fallbacks():
    reg = SpeakerRegistry()
    assert reg.speaker_at(5) == "S1"
    assert reg.speaker_at(5, prev="S7") == "S7"
    reg.add_block(block(0, 60, [T("a", 0, 10), T("b", 10.5, 20), T("a", 25, 60)], {"a": A, "b": B}))
    assert reg.speaker_at(5) == "S1" and reg.speaker_at(15, prev="S1") == "S2"
    assert reg.speaker_at(10.1) == "S1" and reg.speaker_at(10.4) == "S2"   # nearest within 0.5 s
    assert reg.speaker_at(10.25, prev="S1") == "S1"                        # tie: previous speaker
    assert reg.speaker_at(10.25) == "S2"                                   # tie: the turn about to start
    assert reg.speaker_at(22.5, prev="S2") == "S2"                         # gap: keep the previous speaker
    assert reg.speaker_at(22.5) == "S1"                                    # gap, no prev: most talkative
    assert reg.speaker_at(20.6) == "S1"


def test_speaker_at_a_shared_boundary_picks_the_turn_that_starts_there():
    reg = SpeakerRegistry()
    reg.add_block(block(0, 20, [T("a", 0, 10), T("b", 10, 20)], {"a": A, "b": B}))
    assert reg.speaker_at(10) == "S2" and reg.speaker_at(20) == "S2"


# -- reference clips ------------------------------------------------------------------------------

def _podcast():
    """S1 (A) and S2 (B) over 0–200 s with crosstalk, a cold open, and turns of assorted lengths."""
    exclusive = [
        T("a", 10, 20),       # cold open
        T("b", 20, 40),
        T("a", 70, 80),       # clean but B's backchannel sits right before it
        T("b", 80, 95),
        T("a", 100, 104),     # B talks over the middle: nothing clean is left
        T("b", 104, 119.5),   # 0.5 s before A's next turn
        T("a", 120, 150),     # long and clean: split into three
        T("b", 150, 200),
    ]
    crosstalk = [T("b", 69, 70.5), T("b", 101.5, 102)]
    reg = SpeakerRegistry()
    reg.add_block(block(0, 200, exclusive, {"a": A, "b": B}, turns=exclusive + crosstalk))
    return reg


def _check_clean(reg, speaker, clips, min_clip=2.0, max_clip=12.0, guard=0.3):
    own = reg.turns_in(0, 1e9)
    others = [t for t in reg.turns_in(0, 1e9, exclusive=False) + own if t.speaker != speaker]
    for a, b in clips:
        assert min_clip - 1e-9 <= b - a <= max_clip + 1e-9
        assert any(t.speaker == speaker and t.start + guard - 1e-9 <= a and b <= t.end - guard + 1e-9 for t in own)
        for o in others:
            assert o.end + guard <= a + 1e-9 or b + guard <= o.start + 1e-9


def test_reference_clips_skip_crosstalk_and_the_cold_open():
    reg = _podcast()
    clips = reg.reference_clips("S1", target=30)
    _check_clean(reg, "S1", clips)
    # 120–150 → 120.3–149.7 split into three 9.8 s clips, the middle one furthest from B;
    # 70–80 starts only after B's backchannel.
    assert clips[:2] == [pytest.approx((130.1, 139.9)), pytest.approx((120.3, 130.1))]
    assert sorted(clips) == [pytest.approx(c) for c in [(70.8, 79.7), (120.3, 130.1), (130.1, 139.9), (139.9, 149.7)]]
    assert all(a >= 60 for a, _ in clips)
    assert not any(100 <= a < 104 for a, _ in clips)


def test_reference_clips_stop_once_the_target_is_reached():
    reg = _podcast()
    clips = reg.reference_clips("S1", target=10)
    assert len(clips) == 2 and sum(b - a for a, b in clips) >= 10
    assert len(reg.reference_clips("S1", target=5)) == 1
    assert reg.reference_clips("S1", target=0) == []


def test_reference_clips_fall_back_to_the_cold_open_only_to_top_up():
    reg = _podcast()
    clips = reg.reference_clips("S1", target=60)       # everything after 60 s is not enough
    assert clips[-1] == pytest.approx((10.3, 19.7)) and all(a >= 60 for a, _ in clips[:-1])
    reg2 = SpeakerRegistry()
    reg2.add_block(block(0, 60, [T("a", 5, 25), T("b", 30, 55)], {"a": A, "b": B}))
    assert reg2.reference_clips("S1") == [pytest.approx((5.3, 15.0)), pytest.approx((15.0, 24.7))]  # nothing later
    assert reg2.reference_clips("S1", avoid_before=None, target=5) == [pytest.approx((5.3, 15.0))]


def test_reference_clips_prefer_isolated_speech_at_equal_length():
    reg = SpeakerRegistry()
    exclusive = [T("a", 100, 110), T("b", 110.5, 120), T("a", 125, 135), T("b", 140, 150)]
    reg.add_block(block(0, 200, exclusive, {"a": A, "b": B}))
    # Both A clips are 9.4 s after the guard; the second is 5.3 s from B on both sides.
    assert reg.reference_clips("S1", target=5) == [pytest.approx((125.3, 134.7))]
    assert reg.reference_clips("S1", target=20) == [pytest.approx((125.3, 134.7)), pytest.approx((100.3, 109.7))]


def test_reference_clips_respect_min_clip_and_unknown_speakers():
    reg = SpeakerRegistry()
    reg.add_block(block(0, 200, [T("a", 100, 102.5), T("b", 103, 110), T("a", 120, 121.9)], {"a": A, "b": B}))
    assert reg.reference_clips("S1") == []                     # 2.5 s turn is 1.9 s after the guard
    assert reg.reference_clips("S1", min_clip=1.5) == [pytest.approx((100.3, 102.2))]
    assert reg.reference_clips("S9") == []


# -- best span and clean clips --------------------------------------------------------------------

def _stretches(reg, speaker, bridge=0.6):
    """The speaker's exclusive turns joined across pauses of up to `bridge` s (tests cover every gap)."""
    out = []
    for t in reg.turns_in(0, 1e9):
        if t.speaker != speaker:
            continue
        if out and t.start - out[-1][1] <= bridge:
            out[-1] = (out[-1][0], max(out[-1][1], t.end))
        else:
            out.append((t.start, t.end))
    return out


def _check_span(reg, speaker, span, lo, hi, guard=0.3):
    """span is lo–hi s long, inside one of the speaker's stretches and `guard` clear of every other
    speaker's turn."""
    a, b = span
    assert lo - 1e-9 <= b - a <= hi + 1e-9
    assert any(s <= a + 1e-9 and b <= e + 1e-9 for s, e in _stretches(reg, speaker))
    for o in reg.turns_in(0, 1e9, exclusive=False) + reg.turns_in(0, 1e9):
        if o.speaker != speaker:
            assert o.end + guard <= a + 1e-9 or b + guard <= o.start + 1e-9


def _isolation(reg, speaker, span):
    a, b = span
    return min((max(o.start - b, a - o.end) for o in reg.turns_in(0, 1e9, exclusive=False) + reg.turns_in(0, 1e9)
                if o.speaker != speaker), default=math.inf)


def test_best_span_is_one_window_in_the_cleanest_longest_stretch():
    reg = _podcast()
    # 120–149.7 (0.5 s after B: not trimmed further) offers three 10 s windows; the middle one is furthest from B.
    span = reg.best_span("S1")
    assert span == pytest.approx((129.85, 139.85))
    _check_span(reg, "S1", span, 6, 10)
    span = reg.best_span("S1", length=8, min_len=4)
    assert span[1] - span[0] == pytest.approx(8) and _isolation(reg, "S1", span) >= 2 - 1e-9  # a fully isolated 8 s window
    assert reg.best_span("S2") is not None and reg.best_span("S9") is None
    assert reg.best_span("S1", length=0) is None


def test_best_span_bridges_short_pauses_but_never_another_voice():
    exclusive = [T("b", 90, 99),
                 T("a", 100, 103.8), T("a", 104.2, 108.5), T("a", 108.9, 112),   # one breath-paused stretch
                 T("b", 118, 125),
                 T("a", 130, 134), T("a", 134.5, 139),                           # B laughs in the pause
                 T("a", 150, 154.5), T("a", 155.5, 160)]                         # a 1 s pause: two stretches
    laugh = [T("b", 134.1, 134.4)]
    reg = SpeakerRegistry()
    sa = reg.add_block(block(0, 200, exclusive, {"a": A, "b": B}, turns=exclusive + laugh))["a"]
    span = reg.best_span(sa)
    _check_span(reg, sa, span, 6, 10)
    # 100–112 has pauses at 104.0 and 108.7: of the windows that end in one, the one further from B.
    assert span == pytest.approx((104.0, 112.0))
    assert reg.best_span(sa, length=11.4) == pytest.approx((104.0, 112.0))
    reg2 = SpeakerRegistry()
    sa = reg2.add_block(block(0, 200, exclusive[4:], {"a": A, "b": B}, turns=exclusive[4:] + laugh))["a"]
    assert reg2.best_span(sa) is None                       # 130–133.8, 134.7–139, 150–154.5, 155.5–160
    assert reg2.best_span(sa, min_len=4) == pytest.approx((150.0, 154.5))


def test_best_span_prefers_isolation_to_a_little_more_length():
    reg = SpeakerRegistry()
    reg.add_block(block(0, 100, [T("a", 10, 20.6), T("b", 20.6, 30), T("a", 50, 59.9)], {"a": A, "b": B}))
    assert reg.best_span("S1") == pytest.approx((50.0, 59.9))    # 9.9 s, 2 s clear of B, beats 10 s 0.6 s from B


def test_best_span_avoids_the_cold_open_unless_nothing_else_is_left():
    reg = SpeakerRegistry()
    reg.add_block(block(0, 200, [T("a", 0, 30), T("b", 40, 50), T("a", 100, 108), T("b", 108.3, 120)], {"a": A, "b": B}))
    assert reg.best_span("S1") == pytest.approx((0.0, 10.0))
    assert reg.best_span("S1", avoid_before=60) == pytest.approx((100.0, 108.0))
    reg2 = SpeakerRegistry()
    reg2.add_block(block(0, 80, [T("a", 5, 25), T("b", 30, 55), T("a", 55.5, 58)], {"a": A, "b": B}))
    assert reg2.best_span("S1", avoid_before=60) == pytest.approx((5.0, 15.0))  # nothing later
    reg3 = SpeakerRegistry()
    sa = reg3.add_block(block(0, 120, [T("b", 0, 40), T("a", 40.5, 90)], {"a": A, "b": B}))["a"]
    a, b = reg3.best_span(sa, avoid_before=60)             # a stretch across the mark: only its late part counts
    assert a >= 60 and b - a == pytest.approx(10)


def test_best_span_takes_the_callers_score_and_its_rejections():
    reg = _podcast()
    calls = []

    def later_is_better(a, b):
        calls.append((a, b))
        return 1.0 if a > 139 else 0.0

    assert reg.best_span("S1", score=later_is_better) == pytest.approx((139.7, 149.7))
    assert len(calls) == 5                                  # three windows, 70.8–79.7 and the cold open
    music = reg.best_span("S1", avoid_before=60, score=lambda a, b: -math.inf if a > 100 else 0.0)
    assert music == pytest.approx((70.8, 79.7))
    assert reg.best_span("S1", score=lambda a, b: math.nan) is None


def test_best_span_asks_about_a_batch_at_a_time():
    exclusive = []
    for i in range(20):                                     # 20 isolated turns, each 0.5 s shorter than the one before
        exclusive += [T("a", 100 * i, 100 * i + 10 - 0.5 * i), T("b", 100 * i + 50, 100 * i + 60)]
    reg = SpeakerRegistry()
    reg.add_block(block(0, 2000, exclusive, {"a": A, "b": B}))
    calls = []
    assert reg.best_span("S1", length=10, min_len=0.5, score=lambda a, b: calls.append(a) or 0.0) == pytest.approx((0.0, 10.0))
    assert calls == [pytest.approx(100.0 * i) for i in range(8)]   # clear ranks: the best 8
    calls.clear()
    span = reg.best_span("S1", length=10, min_len=0.5, score=lambda a, b: calls.append(a) or (None if len(calls) <= 8 else 0.0))
    assert span == pytest.approx((800.0, 806.0)) and len(calls) == 16  # the first 8 all rejected: the next batch


def test_best_span_spreads_its_questions_over_near_ties():
    """In a long monologue every window scores about the same for length and isolation, so the caller's
    score must be asked about windows across the whole talk, not the first few."""
    exclusive, t = [], 0.0
    while t < 600:                                          # 8 s turns, 0.4 s breaths: one bridged stretch
        exclusive.append(T("a", t, t + 8.0))
        t += 8.4
    reg = SpeakerRegistry()
    sa = reg.add_block(block(0, 600, exclusive, {"a": A}))["a"]
    calls = []

    def snr(a, b):                                          # a music bed under the first 200 s
        calls.append(a)
        return 0.0 if a < 200 else 1.0

    a, b = reg.best_span(sa, avoid_before=60, score=snr)
    assert a >= 200 and len(calls) <= 8
    assert max(calls) - min(calls) > 400                    # spread over the talk


def test_best_span_edges_fall_in_pauses_not_inside_words():
    """A window of a breath-paused stretch starts and ends in a pause (or at the stretch's own ends), so the
    timbre prompt does not open or close mid-word; where silence borders the stretch it is not trimmed."""
    exclusive = [T("b", 0, 20)]
    t = 40.0
    for d in (3.1, 2.7, 3.6, 2.2, 3.3, 2.9, 3.4):            # A's turns, 0.3 s pauses between them
        exclusive.append(T("a", t, t + d))
        t += d + 0.3
    exclusive.append(T("b", t + 5, t + 30))
    reg = SpeakerRegistry()
    sa = reg.add_block(block(0, t + 30, exclusive, {"a": A, "b": B}))["a"]
    own = [x for x in exclusive if x.speaker == "a"]
    for length, min_len in ((10, 6), (8, 4), (6, 3)):
        a, b = reg.best_span(sa, length=length, min_len=min_len)
        _check_span(reg, sa, (a, b), min_len, length)
        for edge in (a, b):
            assert not any(x.start + 1e-9 < edge < x.end - 1e-9 for x in own)
    assert reg.best_span(sa, length=30)[0] == pytest.approx(40.0)   # the stretch starts at A's first word


def test_clean_clips_cover_the_clean_speech_best_first_up_to_the_budget():
    reg = _podcast()
    clips = reg.clean_clips("S1")
    for c in clips:
        _check_span(reg, "S1", c, 2.0, 10.0)
    # The cold open starts at A's first word (silence before it); 120–149.7 splits into three 9.9 s clips.
    assert sorted(clips) == [pytest.approx(c) for c in
                             [(10.0, 19.7), (70.8, 79.7), (120.0, 129.9), (129.9, 139.8), (139.8, 149.7)]]
    assert clips[:2] == [pytest.approx((129.9, 139.8)), pytest.approx((120.0, 129.9))]
    assert reg.clean_clips("S1", max_total=20) == clips[:2]      # 0.2 s of room left: under min_clip
    assert reg.clean_clips("S1", max_total=25)[2] == pytest.approx((12.25, 17.45))  # trimmed about its middle
    assert sum(b - a for a, b in reg.clean_clips("S1", max_total=25)) == pytest.approx(25)
    assert reg.clean_clips("S1", max_total=25, avoid_before=60)[2] == pytest.approx((72.65, 77.85))
    assert reg.clean_clips("S1", min_clip=9.75) == [pytest.approx((129.9, 139.8)), pytest.approx((120.0, 129.9)),
                                                    pytest.approx((139.8, 149.7))]
    assert reg.clean_clips("S9") == [] and reg.clean_clips("S1", max_total=1) == []


@settings(max_examples=150, deadline=None)
@given(st.lists(st.tuples(st.sampled_from("ab"), st.floats(0.2, 15), st.floats(0.0, 2.0)), min_size=1, max_size=40),
       st.lists(st.tuples(st.floats(0, 1), st.floats(0.1, 1.0)), max_size=6),
       st.floats(4, 12), st.floats(1, 6), st.one_of(st.none(), st.floats(0, 100)))
def test_best_span_and_clean_clips_stay_clean(spec, cross, length, min_len, avoid):
    exclusive, t = [], 0.0
    for who, dur, gap in spec:
        exclusive.append(T(who, t, t + dur))
        t += dur + gap
    crosstalk = [T("b", f * t, f * t + d) for f, d in cross]  # B talking over anyone, A included
    reg = SpeakerRegistry()
    sa = reg.add_block(block(0, t + 1, exclusive, {"a": A, "b": B}, turns=exclusive + crosstalk)).get("a")
    if sa is None:
        return                                              # A said too little to become a speaker
    span = reg.best_span(sa, length=length, min_len=min_len, avoid_before=avoid)
    if span is not None:
        _check_span(reg, sa, span, min(min_len, length), length)
    clips = reg.clean_clips(sa, max_total=30, min_clip=min_len)
    for c in clips:
        _check_span(reg, sa, c, min_len, 10)
    assert sum(b - a for a, b in clips) <= 30 + 1e-9
    assert all(x[1] <= y[0] + 1e-9 or y[1] <= x[0] + 1e-9 for i, x in enumerate(clips) for y in clips[i + 1:])
    if clips and avoid is None and min_len <= length:
        assert span is not None                             # a clean piece >= min_len means a stretch that long


# -- property: stitching blocks recovers the true timeline ----------------------------------------

SR = 16_000
PDIM = 1024
timeline = st.lists(st.tuples(st.integers(0, 2), st.floats(0.5, 25), st.floats(0.0, 3.0)), min_size=6, max_size=80)


def _embedding(toward, sim, noise):
    """A unit embedding with cosine exactly `sim` to the unit vector `toward`, the rest along e_noise, a
    direction no other embedding uses. Each person's vectors live on their own axes, so embeddings of
    different people have cosine 0 and the one to a person's centroid is exactly what the test picked."""
    v = sim * toward
    v[noise] = math.sqrt(1 - sim * sim)
    return v


@settings(max_examples=150, deadline=None)
@given(timeline, st.floats(25, 120), st.floats(0, 15), st.integers(0, 2**16), st.booleans())
def test_blocks_stitch_back_into_the_true_timeline(spec, block_len, overlap, seed, seek):
    """Blocks cut the way the session cuts them (ends on whole samples; with `seek`, one later block first
    and the skipped ones after it), labels renamed per block, embeddings from just above the threshold
    up, or below it wherever >= 2 s of shared overlap speech vouches for them, and short split-off
    fragments whose embedding beats the main label's."""
    rng = np.random.default_rng(seed)
    truth, t, prev = [T("P0", 0, 4), T("P1", 4.5, 9), T("P2", 9.5, 14)], 14.5, 2   # everyone speaks up in block 1
    for who, length, gap in spec:
        if who == prev:
            who = (who + 1) % 3
        truth.append(T(f"P{who}", t, t + length))
        t, prev = t + length + gap, who
    end = truth[-1].end

    cuts, a = [], 0.0
    while True:
        b = min(a + block_len, end)
        cuts.append((a, a + (int(b * SR) - int(a * SR)) / SR))  # the audio slice the diarizer actually gets
        if b >= end:
            break
        a = b - overlap
    if seek and len(cuts) > 3:
        k = int(rng.integers(2, len(cuts)))
        cuts = [cuts[0], cuts[k], *cuts[1:k], *cuts[k + 1:]]

    reg, noise, axes, gid = SpeakerRegistry(), iter(range(3, PDIM)), np.eye(PDIM), {}
    for a, b in cuts:
        names = {p: f"SPEAKER_{i:02d}" for i, p in enumerate(rng.permutation(["P0", "P1", "P2"]))}
        person = {lab: int(p[1]) for p, lab in names.items()}
        excl = [T(names[x.speaker], max(x.start, a), min(x.end, b)) for x in truth if x.end > a and x.start < b]
        long, split = [i for i, x in enumerate(excl) if x.end - x.start >= 4], None
        if long and rng.random() < 0.5:
            i = long[int(rng.integers(len(long)))]
            x, f = excl[i], float(rng.uniform(0.2, 1.5))
            m = (x.start + x.end) / 2
            excl[i:i + 1] = [T(x.speaker, x.start, m), T("FRAG", m, m + f), T(x.speaker, m + f, x.end)]
            split, person["FRAG"] = x.speaker, person[x.speaker]
        known = reg.turns_in(a, b)
        cents = {}
        for lab in sorted({x.speaker for x in excl}):
            p = person[lab]
            toward = reg.speakers[gid[p]].centroid if p in gid else axes[p]  # cosines are to the current centroid
            shared = sum(max(0.0, min(x.end, k.end) - max(x.start, k.start)) for x in excl if x.speaker == lab for k in known)
            if lab == "FRAG":
                lo, hi = 0.95, 0.99   # beats its main label (it follows a centroid the main label just moved: >= 0.95 * 0.6)
            elif shared >= 2.5 and lab != split and rng.random() < 0.5:
                lo, hi = 0.32, 0.49   # drifted below the threshold: only the shared speech can link it
            else:
                lo, hi = (0.6 if lab == split else 0.55), 0.95
            cents[lab] = _embedding(toward, float(rng.uniform(lo, hi)), next(noise))
        for lab, g in reg.add_block(block(a, b, excl, cents)).items():
            if lab != "FRAG":
                gid[person[lab]] = g

    got = reg.turns_in(0, end)
    for x, y in zip(got, got[1:]):
        assert x.end <= y.start + 1e-9                          # exclusive turns never overlap
    ids: dict[str, str] = {}
    for x in truth:
        pieces = reg.turns_in(x.start, x.end)
        assert len({p.speaker for p in pieces}) == 1
        assert ids.setdefault(x.speaker, pieces[0].speaker) == pieces[0].speaker
        assert sum(p.end - p.start for p in pieces) == pytest.approx(x.end - x.start, abs=0.05)
    assert len(set(ids.values())) == len(ids) == len(reg.speakers)  # one global id per real person
    assert reg.covered_until() == pytest.approx(end, abs=1e-3) and reg.is_covered(0, end)  # ends on whole samples
    total = sum(g.talk_seconds for g in reg.speakers.values())
    assert total == pytest.approx(sum(x.end - x.start for x in truth), abs=0.05 * len(truth))
