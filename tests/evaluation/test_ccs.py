import numpy as np
import pytest

from evaluation.ccs import calculate_ccs, mutate_base


def _predict_fn_2d(sgrnas: list[str], off_seqs: list[str]) -> np.ndarray:
    probs = []
    for guide, off in zip(sgrnas, off_seqs):
        spacer = off[:20]
        pam = off[20:23]

        # Penalita posizionale: mismatch PAM-proximal (indice alto) pesa di piu.
        weights = np.linspace(0.01, 0.05, 20)
        mismatch_penalty = sum(weights[i] for i, (g, t) in enumerate(zip(guide, spacer)) if g != t)

        pam_adjust = {
            "AGG": 0.0,
            "AAG": -0.20,
            "ACG": -0.30,
            "AAA": -0.40,
        }.get(pam, -0.25)

        p = float(np.clip(0.95 - mismatch_penalty + pam_adjust, 1e-4, 1.0 - 1e-4))
        probs.append(p)

    p1 = np.asarray(probs, dtype=float)
    return np.column_stack([1.0 - p1, p1])


def test_calculate_ccs_accepts_2d_predict_output() -> None:
    guides = ["ACGTACGTACGTACGTACGT", "TTTTCCCCAAAAGGGGTTTT"]
    out = calculate_ccs(guides, _predict_fn_2d, mode="3_rules")

    assert "CCS_Overall" in out
    assert 0.0 <= out["CCS_Overall"] <= 1.0
    assert all(0.0 <= float(v) <= 1.0 for v in out.values())


def test_mutate_base_wobble_is_distinct_for_ac() -> None:
    assert mutate_base("A", "wobble") == "G"
    assert mutate_base("A", "wobble") != mutate_base("A", "transversion")
    assert mutate_base("C", "wobble") == "T"
    assert mutate_base("C", "wobble") != mutate_base("C", "transversion")


def test_calculate_ccs_rejects_empty_guides() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_ccs([], _predict_fn_2d)


def test_calculate_ccs_rejects_short_guides() -> None:
    with pytest.raises(ValueError, match="at least 20 nt"):
        calculate_ccs(["ACGT"], _predict_fn_2d)
