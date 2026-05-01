from __future__ import annotations

import itertools
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from dag.mismatch import classify_mismatch


class BaseEncoder(nn.Module, ABC):
    """
    Interfaccia base per tutti gli encoder di sequenze CRISPR.
    """

    @abstractmethod
    def encode(self, sgrnas: list[str], off_targets: list[str]) -> torch.Tensor:
        """
        Codifica la regione spacer (pairwise).
        Restituisce un tensore di shape (batch_size, 20, embed_dim).
        """
        pass

    @abstractmethod
    def encode_pam(self, off_targets: list[str]) -> torch.Tensor:
        """
        Codifica la regione PAM.
        Restituisce un tensore di shape (batch_size, 3, embed_dim).
        """
        pass

    def forward(self, sgrnas: list[str], off_targets: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Restituisce la tupla (spacer_encoded, pam_encoded).
        """
        return self.encode(sgrnas, off_targets), self.encode_pam(off_targets)


class PairwiseTokenEncoder(BaseEncoder):
    """
    Costruisce un vocabolario di coppie di basi (sgRNA_base, target_base).
    Supporta l'alfabeto esteso {A, C, G, T, N}, generando 25 token unici.
    """

    ALPHABET = ["A", "C", "G", "T", "N"]
    
    def __init__(self, embed_dim: int = 16, use_learned_embeddings: bool = True):
        super().__init__()
        
        self.use_learned_embeddings = use_learned_embeddings
        self.vocab_size = len(self.ALPHABET) ** 2  # 5 * 5 = 25
        
        # Mappatura statica (base1, base2) -> intero
        self._token_map: dict[tuple[str, str], int] = {
            pair: idx for idx, pair in enumerate(itertools.product(self.ALPHABET, repeat=2))
        }
        self._fallback_token = self._token_map[("N", "N")]
        self._pam_token_map: dict[str, int] = {base: idx for idx, base in enumerate(self.ALPHABET)}
        self._pam_fallback_token = self._pam_token_map["N"]

        if self.use_learned_embeddings:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(
                num_embeddings=self.vocab_size, 
                embedding_dim=self.embed_dim
            )
            self.pam_embedding = nn.Embedding(
                num_embeddings=len(self.ALPHABET),
                embedding_dim=self.embed_dim,
            )
            nn.init.xavier_uniform_(self.embedding.weight)
            nn.init.xavier_uniform_(self.pam_embedding.weight)
        else:
            self.embed_dim = self.vocab_size

    def _tokenize_pair(self, sgrna: str, off_target: str) -> list[int]:
        """Converte una singola coppia di sequenze in una lista di token (interi)."""
        tokens = []
        for b1, b2 in zip(sgrna.upper(), off_target.upper()):
            token = self._token_map.get((b1, b2), self._fallback_token)
            tokens.append(token)
        return tokens

    def _tokenize_pam(self, pam: str) -> list[int]:
        """Converte una sequenza PAM in token singoli su vocabolario dedicato."""
        return [self._pam_token_map.get(base, self._pam_fallback_token) for base in pam.upper()]

    def encode(self, sgrnas: list[str], off_targets: list[str]) -> torch.Tensor:
        """
        Esegue la codifica in batch della regione spacer (posizioni 0-19).
        Restituisce: Tensor di shape (B, 20, embed_dim).
        """
        if len(sgrnas) != len(off_targets):
            raise ValueError("Le liste sgrnas e off_targets devono avere la stessa lunghezza batch.")
        if not sgrnas:
            raise ValueError("Input batch vuoto.")

        # Isoliamo rigorosamente lo spacer (primi 20 nucleotidi) e facciamo padding
        spacer_sgrnas = [s[:20].ljust(20, "N") for s in sgrnas]
        spacer_targets = [t[:20].ljust(20, "N") for t in off_targets]

        batch_tokens = [self._tokenize_pair(s, t) for s, t in zip(spacer_sgrnas, spacer_targets)]
        
        device = self.embedding.weight.device if self.use_learned_embeddings else torch.device("cpu")
        token_tensor = torch.tensor(batch_tokens, dtype=torch.long, device=device)

        if self.use_learned_embeddings:
            encoded = self.embedding(token_tensor)
        else:
            encoded = F.one_hot(token_tensor, num_classes=self.vocab_size).float()

        return encoded

    def encode_pam(self, off_targets: list[str]) -> torch.Tensor:
        """
        Esegue la codifica in batch della regione PAM (posizioni 20-22).
        Usa un vocabolario dedicato a 5 token (A/C/G/T/N), evitando la sparsezza
        del vocabolario pairwise usato dallo spacer.
        Restituisce: Tensor di shape (B, 3, embed_dim).
        """
        # Estraiamo il PAM o usiamo NNN come fallback se mancante
        pam_seqs = [t[20:23] if len(t) >= 23 else "NNN" for t in off_targets]
        
        # Allineamento PAM a 3 caratteri per sicurezza
        pam_seqs = [p.ljust(3, "N")[:3] for p in pam_seqs]

        # Tokenizzazione base-per-base su vocabolario PAM dedicato
        batch_tokens = [self._tokenize_pam(p) for p in pam_seqs]
        
        if self.use_learned_embeddings:
            device = self.pam_embedding.weight.device
        else:
            device = torch.device("cpu")
        token_tensor = torch.tensor(batch_tokens, dtype=torch.long, device=device)

        if self.use_learned_embeddings:
            encoded = self.pam_embedding(token_tensor)
        else:
            encoded = F.one_hot(token_tensor, num_classes=len(self.ALPHABET)).float()

        return encoded


class BiologicalMismatchEncoder(BaseEncoder):
    """
    Encoding esplicito biologicamente motivato.
    Per ogni posizione produce un vettore con:
    - tipo di mismatch (4 classi one-hot: match/wobble/transition/transversion)
    - base sgRNA (4 classi one-hot: A/C/G/T)
    - base target (4 classi one-hot: A/C/G/T)
    Output: Tensor[B, 20, 12] — no parametri learnable nello spacer
    """

    MISMATCH_TYPES = {
        "match": 0,
        "wobble": 1,
        "transition": 2,
        "transversion": 3,
    }
    BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 0}

    def __init__(self):
        super().__init__()
        self.embed_dim = 12  # fisso, no parametri
        # Registra un buffer dummy per tracciare il device quando model.to(device) è chiamato
        self.register_buffer("_device_tracker", torch.zeros(1))

    def encode(self, sgrnas: list[str], off_targets: list[str]) -> torch.Tensor:
        """
        Encoding biologicamente motivato dello spacer (20 bp).
        Restituisce: Tensor[B, 20, 12] con one-hot concatenati.
        """
        B = len(sgrnas)
        device = self._device_tracker.device
        out = torch.zeros(B, 20, 12, dtype=torch.float32, device=device)

        for b, (sg, ot) in enumerate(zip(sgrnas, off_targets)):
            sg = sg[:20].ljust(20, "N")
            ot = ot[:20].ljust(20, "N")

            for i, (bg, bt) in enumerate(zip(sg, ot)):
                mt = classify_mismatch(bg, bt)  # restituisce MismatchType string

                # One-hot tipo mismatch (4 dim)
                out[b, i, self.MISMATCH_TYPES[mt]] = 1.0
                # One-hot base sgRNA (4 dim)
                out[b, i, 4 + self.BASE_IDX[bg.upper()]] = 1.0
                # One-hot base target (4 dim)
                out[b, i, 8 + self.BASE_IDX[bt.upper()]] = 1.0

        return out

    def encode_pam(self, off_targets: list[str]) -> torch.Tensor:
        """
        Encoding PAM (3 bp): 5 classi one-hot (A/C/G/T/N) paddate a embed_dim=12.
        Restituisce: Tensor[B, 3, 12] per coerenza con lo spacer encoding.
        """
        B = len(off_targets)
        device = self._device_tracker.device
        # Crea output (B, 3, 12) — i primi 5 dim sono one-hot, il resto è zero padding
        out = torch.zeros(B, 3, 12, dtype=torch.float32, device=device)
        BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

        for b, ot in enumerate(off_targets):
            pam = (ot[20:23] if len(ot) >= 23 else "NNN").ljust(3, "N")
            for i, base in enumerate(pam):
                out[b, i, BASE_IDX.get(base.upper(), 4)] = 1.0

        return out