from dataclasses import dataclass, field

import numpy as np

from dag.mismatch import build_mismatch_vectors, gc_content
from dag.pam import extract_pam_from_offtarget, pam_score


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@dataclass(init=False, slots=True)
class CRISPRPairFeatures:
    """DAG node container for one sgRNA/off-target pair."""

    sgRNA_seq: str
    off_seq: str

    sgRNA_spacer: str = field(init=False)
    off_spacer: str = field(init=False)
    pam: str = field(init=False)

    mm_vector: np.ndarray = field(init=False, repr=False)
    type_vector: np.ndarray = field(init=False, repr=False)
    energy_vector: np.ndarray = field(init=False, repr=False)

    def __init__(
        self,
        sgRNA_seq: str | None = None,
        off_seq: str | None = None,
        *,
        guide_seq: str | None = None,
        target_seq: str | None = None,
        pam: str | None = None,
        enzyme: str | None = None,
    ) -> None:
        # Legacy constructor support: guide_seq/target_seq/pam.
        if sgRNA_seq is None:
            sgRNA_seq = guide_seq

        if off_seq is None and target_seq is not None:
            target_clean = target_seq.upper().replace("U", "T")
            if len(target_clean) >= 23:
                off_seq = target_clean[:23]
            elif pam is not None and len(target_clean) >= 20:
                off_seq = target_clean[:20] + pam.upper().replace("U", "T")

        if sgRNA_seq is None or off_seq is None:
            raise ValueError("CRISPRPairFeatures requires sgRNA_seq and off_seq")

        self.sgRNA_seq = sgRNA_seq.upper().replace("U", "T")
        self.off_seq = off_seq.upper().replace("U", "T")
        self._initialize_vectors()

    def _initialize_vectors(self) -> None:
        self.sgRNA_spacer = self.sgRNA_seq[:20].ljust(20, "N")
        off_full = self.off_seq if len(self.off_seq) >= 23 else self.off_seq.ljust(23, "N")
        self.off_spacer = off_full[:20]
        self.pam = extract_pam_from_offtarget(off_full)
        self.mm_vector, self.type_vector, self.energy_vector = build_mismatch_vectors(
            self.sgRNA_spacer,
            self.off_spacer,
        )

    @property
    def node_A_pam(self) -> float:
        return _clip01(pam_score(self.pam))

    @property
    def node_B_proximal(self) -> float:
        # Posizioni 1-4 PAM-proximal, indice 0 corrisponde alla posizione 1.
        return _clip01(float(np.mean(self.energy_vector[0:4])))

    @property
    def node_C_seed_extension(self) -> float:
        # Posizioni 5-12.
        return _clip01(float(np.mean(self.energy_vector[4:12])))

    @property
    def node_D_non_seed(self) -> float:
        # Posizioni 13-20.
        return _clip01(float(np.mean(self.energy_vector[12:20])))

    @property
    def guide_length(self) -> int:
        return len(self.sgRNA_spacer)

    @property
    def guide_seq(self) -> str:
        return self.sgRNA_spacer

    @property
    def target_seq(self) -> str:
        return self.off_spacer

    def to_feature_dict(self) -> dict[str, float | str | int]:
        profile = self.to_position_profile()
        feature_dict: dict[str, float | str | int] = {
            "sgRNA_seq": self.sgRNA_seq,
            "off_seq": self.off_seq,
            "sgRNA_spacer": self.sgRNA_spacer,
            "off_spacer": self.off_spacer,
            "pam": self.pam,
            "pam_score": self.node_A_pam,
            "node_A_pam": self.node_A_pam,
            "node_B_proximal": self.node_B_proximal,
            "node_C_seed_extension": self.node_C_seed_extension,
            "node_D_non_seed": self.node_D_non_seed,
            "mismatch_count": int(np.sum(self.mm_vector)),
            "mismatch_rate": float(np.mean(self.mm_vector)),
            "gc_sgRNA": gc_content(self.sgRNA_spacer),
            "gc_offtarget": gc_content(self.off_spacer),
            "mean_energy_penalty": float(np.mean(self.energy_vector)),
            "total_energy_penalty": float(np.sum(self.energy_vector)),
        }
        for idx, value in enumerate(profile, start=1):
            feature_dict[f"profile_pos_{idx:02d}"] = float(value)
        return feature_dict

    def to_concept_dict(self) -> dict[str, float]:
        # 7 concetti normalizzati per CBM.
        concepts = {
            "concept_pam": self.node_A_pam,
            "concept_proximal_load": self.node_B_proximal,
            "concept_seed_extension_load": self.node_C_seed_extension,
            "concept_non_seed_load": self.node_D_non_seed,
            "concept_mismatch_rate": _clip01(float(np.mean(self.mm_vector))),
            "concept_energy": _clip01(float(np.mean(self.energy_vector))),
            "concept_gc_delta": _clip01(abs(gc_content(self.sgRNA_spacer) - gc_content(self.off_spacer))),
        }
        return {key: _clip01(value) for key, value in concepts.items()}

    def to_position_profile(self) -> np.ndarray:
        return np.asarray(self.energy_vector, dtype=float).reshape(20)
