from typing import Literal

import numpy as np

MismatchType = Literal["match", "wobble", "transition", "transversion"]

TRANSITIONS = {
    ("A", "G"),
    ("G", "A"),
    ("C", "T"),
    ("T", "C"),
}

WOBBLE = {
    ("G", "T"),
    ("T", "G"),
}

MISMATCH_ENERGY_PENALTY: dict[MismatchType, float] = {
    "match": 0.00,
    "wobble": 0.25,
    "transition": 0.55,
    "transversion": 0.80,
}

TYPE_TO_INDEX: dict[MismatchType, int] = {
    "match": 0,
    "wobble": 1,
    "transition": 2,
    "transversion": 3,
}


def classify_mismatch(base_g: str, base_t: str) -> MismatchType:
    pair = (base_g.upper(), base_t.upper())
    if pair[0] == pair[1]:
        return "match"
    if pair in WOBBLE:
        return "wobble"
    if pair in TRANSITIONS:
        return "transition"
    return "transversion"


def energy_penalty(mismatch_type: MismatchType) -> float:
    return float(MISMATCH_ENERGY_PENALTY[mismatch_type])


def _normalize_spacer(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    seq = "".join(base if base in {"A", "C", "G", "T", "N"} else "N" for base in seq)
    if len(seq) >= 20:
        return seq[:20]
    return seq.ljust(20, "N")


def build_mismatch_vectors(spacer_g: str, spacer_t: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mm_vector, type_vector, energy_vector), each with shape [20].

    Index 0 corresponds to position 1 (PAM-proximal), so vectors are built from
    the 3' end of the spacer toward the 5' end.
    """
    guide = _normalize_spacer(spacer_g)
    target = _normalize_spacer(spacer_t)

    mm_vector = np.zeros(20, dtype=np.int8)
    type_vector = np.zeros(20, dtype=np.int8)
    energy_vector = np.zeros(20, dtype=float)

    for pos in range(20):
        # Reverse indexing: PAM-proximal base first.
        kind = classify_mismatch(guide[19 - pos], target[19 - pos])
        mm_vector[pos] = 0 if kind == "match" else 1
        type_vector[pos] = TYPE_TO_INDEX[kind]
        energy_vector[pos] = energy_penalty(kind)

    return mm_vector, type_vector, energy_vector


def gc_content(seq: str) -> float:
    normalized = _normalize_spacer(seq)
    if not normalized:
        return 0.0
    gc_count = normalized.count("G") + normalized.count("C")
    return float(gc_count / len(normalized))


# Backward compatibility aliases for existing code/tests.
def mismatch_type(guide_nt: str, target_nt: str) -> str:
    return classify_mismatch(guide_nt, target_nt)


def mismatch_energy_penalty(kind: str, position: int = 0) -> float:
    base = energy_penalty(kind)  # type: ignore[arg-type]
    if kind == "match":
        return 0.0
    seed_multiplier = 1.2 if position < 10 else 1.0
    return float(min(1.0, base * seed_multiplier))


__all__ = [
    "MISMATCH_ENERGY_PENALTY",
    "build_mismatch_vectors",
    "classify_mismatch",
    "energy_penalty",
    "gc_content",
]
