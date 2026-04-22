from itertools import product

NUCLEOTIDES = "ACGT"


def _build_pam_compatibility() -> dict[str, float]:
    """Create continuous compatibility scores for all 64 trinucleotide PAMs."""
    scores: dict[str, float] = {}
    for triplet in ("".join(parts) for parts in product(NUCLEOTIDES, repeat=3)):
        # Heuristic soft gate centered on the NGG motif family.
        score = 0.05
        score += 0.35 if triplet[1] == "G" else 0.10 if triplet[1] == "A" else 0.0
        score += 0.50 if triplet[2] == "G" else 0.20 if triplet[2] == "A" else 0.05
        score += 0.05 if triplet[0] in {"C", "G"} else 0.0
        scores[triplet] = round(max(0.0, min(1.0, score)), 4)

    # Explicit values for common SpCas9-compatible PAMs.
    for prefix in NUCLEOTIDES:
        scores[f"{prefix}GG"] = 1.0 if prefix in {"A", "C", "T"} else 0.98
        scores[f"{prefix}AG"] = 0.65
    return scores


PAM_COMPATIBILITY: dict[str, float] = _build_pam_compatibility()


def extract_pam_from_offtarget(off_seq: str) -> str:
    """Extract the last 3 nt from an off-target sequence."""
    seq = off_seq.upper().replace("U", "T")
    if len(seq) < 3:
        raise ValueError("off_seq must contain at least 3 nucleotides")
    return seq[-3:]


def pam_score(pam: str, enzyme: str | None = None) -> float:
    """Soft PAM compatibility score in [0, 1]."""
    if enzyme is not None and enzyme != "SpCas9":
        return 0.0
    seq = pam.upper().replace("U", "T")
    if len(seq) != 3:
        return 0.0
    return float(PAM_COMPATIBILITY.get(seq, 0.0))


def is_canonical_pam(pam: str, threshold: float = 0.9) -> bool:
    return pam_score(pam) >= threshold


__all__ = [
    "PAM_COMPATIBILITY",
    "extract_pam_from_offtarget",
    "is_canonical_pam",
    "pam_score",
]
