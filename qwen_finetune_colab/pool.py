"""
Frame pooling per SFT_FRAME_POOLING.md: human interval → snippet set I → (frame, score) rows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


def L(i: int, F: int, T: int) -> int:
    if T <= 0:
        return 0
    return int(i * F / T)


def R(i: int, F: int, T: int) -> int:
    if T <= 0:
        return max(0, F - 1)
    nxt = int((i + 1) * F / T)
    lo = L(i, F, T)
    hi = min(F - 1, nxt - 1)
    return lo if hi < lo else hi


def snippet_intersects_human(i: int, a0: int, a1: int, F: int, T: int) -> bool:
    lo, hi = L(i, F, T), R(i, F, T)
    return not (hi < a0 or lo > a1)


def snippets_in_human_window(a0: int, a1: int, F: int, T: int) -> list[int]:
    return [i for i in range(T) if snippet_intersects_human(i, a0, a1, F, T)]


def fallback_snippet_closest_to_interval(a0: int, a1: int, F: int, T: int) -> list[int]:
    """Single snippet whose [L,R] is closest to the human interval (by center distance)."""
    if T <= 0 or F <= 0:
        return []
    center = (a0 + a1) // 2
    best_i, best_d = 0, 10**18
    for i in range(T):
        lo, hi = L(i, F, T), R(i, F, T)
        if lo <= center <= hi:
            d = 0
        elif hi < center:
            d = center - hi
        else:
            d = lo - center
        if d < best_d:
            best_d, best_i = d, i
    return [best_i]


def dedup_frames_preserve_time_order(pairs: Sequence[tuple[int, float]]) -> list[tuple[int, float]]:
    seen: set[int] = set()
    out: list[tuple[int, float]] = []
    for f, s in sorted(pairs, key=lambda x: x[0]):
        if f in seen:
            continue
        seen.add(f)
        out.append((f, s))
    return out


def _mid_frame(i: int, F: int, T: int) -> int:
    lo, hi = L(i, F, T), R(i, F, T)
    return (lo + hi) // 2


def every_snippet_mid(I: Sequence[int], scores: Sequence[float], F: int, T: int) -> list[tuple[int, float]]:
    out = [(_mid_frame(i, F, T), float(scores[i])) for i in I]
    return dedup_frames_preserve_time_order(out)


def every_snippet_first(I: Sequence[int], scores: Sequence[float], F: int, T: int) -> list[tuple[int, float]]:
    out = [(L(i, F, T), float(scores[i])) for i in I]
    return dedup_frames_preserve_time_order(out)


def every_snippet_mid_frame_band(
    I: Sequence[int],
    scores: Sequence[float],
    F: int,
    T: int,
    *,
    delta: int = 2,
) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for i in I:
        lo, hi = L(i, F, T), R(i, F, T)
        m = (lo + hi) // 2
        for d in range(-delta, delta + 1):
            fr = m + d
            if lo <= fr <= hi:
                out.append((fr, float(scores[i])))
    return dedup_frames_preserve_time_order(out)


def human_span_smart(
    I: Sequence[int],
    scores: Sequence[float],
    F: int,
    T: int,
    *,
    budget: int,
    min_gap: int,
) -> list[tuple[int, float]]:
    if not I:
        return []
    I_list = sorted(set(I))
    if len(I_list) == 1:
        i0 = I_list[0]
        return [(_mid_frame(i0, F, T), float(scores[i0]))]

    lo_i, hi_i = I_list[0], I_list[-1]
    selected: set[int] = {lo_i, hi_i}
    interior = [i for i in I_list if i not in selected]
    interior.sort(key=lambda i: (-scores[i], i))

    def ok_gap(idx: int) -> bool:
        return all(abs(idx - j) >= min_gap for j in selected)

    while len(selected) < budget and interior:
        added = False
        for idx in list(interior):
            if len(selected) >= budget:
                break
            if ok_gap(idx):
                selected.add(idx)
                interior.remove(idx)
                added = True
                break
        if not added:
            break

    ordered_snips = sorted(selected)
    out = [(_mid_frame(i, F, T), float(scores[i])) for i in ordered_snips]
    return dedup_frames_preserve_time_order(out)


def top3_snippets_mid_frame_band(
    I: Sequence[int],
    scores: Sequence[float],
    F: int,
    T: int,
    *,
    delta: int = 2,
) -> list[tuple[int, float]]:
    if not I:
        return []
    ranked = sorted(I, key=lambda i: (-scores[i], i))
    k = min(3, len(ranked))
    top = ranked[:k]
    out: list[tuple[int, float]] = []
    for i in top:
        lo, hi = L(i, F, T), R(i, F, T)
        m = (lo + hi) // 2
        block: list[tuple[int, float]] = []
        for d in range(-delta, delta + 1):
            fr = m + d
            if lo <= fr <= hi:
                block.append((fr, float(scores[i])))
        block.sort(key=lambda x: x[0])
        out.extend(block)
    return dedup_frames_preserve_time_order(out)


StrategyFn = Callable[..., list[tuple[int, float]]]

STRATEGIES: dict[str, StrategyFn] = {
    "every_snippet_mid": every_snippet_mid,
    "every_snippet_first": every_snippet_first,
    "every_snippet_mid_frame_band": every_snippet_mid_frame_band,
    "human_span_smart": human_span_smart,
    "top3_snippets_mid_frame_band": top3_snippets_mid_frame_band,
}
