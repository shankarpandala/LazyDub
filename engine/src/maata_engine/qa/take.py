"""Choosing among a line's takes (ARCHITECTURE §3.8, §3.13) while no Telugu recogniser is approved (decision D9).

Each take is judged by what is known without hearing it: a take that ran to its token cap without an end token failed
(the decode ran away), and one whose natural duration is under 0.6 x the estimate for its wording most likely dropped
words. Of the rest, the one whose duration is closest to the line's speech time is voiced.

Step 6's take QA plugs in here: its verdicts (CER against `spoken`, the WeSpeaker similarity outlier guard at matched
length) are more failure reasons per take, added to what `failure` finds before `pick` chooses. Similarity isn't
scored yet: it needs every take vocoded, which the voicer now spares all but the chosen one.
"""

from __future__ import annotations

import math
from typing import Sequence

SHORT = 0.6  # a take under this share of its wording's predicted duration is flagged (§3.13)
# How bad a failure is, when every take failed and the least bad must still be voiced (a line is never skipped
# silently): a short take may only have dropped a word; a take that ran to its cap is a runaway. Reasons QA adds later
# rank with the short ones.
SEVERITY = {None: 0, "short": 1, "cap": 2}


def failure(seconds: float, capped: bool, estimate: float) -> str | None:
    """Why a take failed, or None: "cap" (it ran to its token cap without ending) or "short" (under 0.6 x `estimate`,
    the predicted duration of its wording)."""
    if capped:
        return "cap"
    if seconds < SHORT * estimate:
        return "short"
    return None


def pick(seconds: Sequence[float], failures: Sequence[str | None], target: float) -> int:
    """The take to voice: of those that didn't fail, the one whose natural duration is closest to `target` (the line's
    speech time) by ratio, the first on a tie; when all failed, the least bad the same way."""
    def fit(k: int) -> tuple[int, float, int]:
        ratio = max(seconds[k], 1e-3) / max(target, 1e-3)
        return SEVERITY.get(failures[k], 1), abs(math.log(ratio)), k

    return min(range(len(seconds)), key=fit)
