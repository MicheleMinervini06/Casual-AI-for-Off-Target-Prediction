from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import PairwiseTokenEncoder, BiologicalMismatchEncoder, BaseEncoder
from .modules import NonSeedModule, PAMModule, ProximalModule, SeedExtensionModule


class NeuralSCM(nn.Module):
    """
    Neural Structural Causal Model per previsioni CRISPR.
    Assembla moduli indipendenti in un DAG causale esplicito.
    """

    def __init__(self, embed_dim: int = 16, hidden_dim: int = 32, encoder: BaseEncoder | None = None):
        super().__init__()
        
        # Se nessun encoder è fornito, usa PairwiseTokenEncoder di default
        if encoder is None:
            encoder = PairwiseTokenEncoder(embed_dim=embed_dim)
        
        self.encoder = encoder
        self.embed_dim = encoder.embed_dim  # Adatta embed_dim all'encoder

        self.pam_node = PAMModule(embed_dim=encoder.embed_dim, hidden_dim=hidden_dim)
        self.proximal_node = ProximalModule(embed_dim=encoder.embed_dim)
        self.seed_node = SeedExtensionModule(embed_dim=encoder.embed_dim)
        self.nonseed_node = NonSeedModule(embed_dim=encoder.embed_dim)

        self.w_proximal = nn.Parameter(torch.randn(1))
        self.w_seed = nn.Parameter(torch.randn(1))
        self.w_nonseed = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def _base_forward(
        self, 
        sgrnas: list[str], 
        off_targets: list[str], 
        intervention: dict[str, float] | None = None
    ) -> dict[str, torch.Tensor]:
        
        if intervention is None:
            intervention = {}

        # Lasciamo l'encoder attivo solo perché ci serve il device e per il PAM
        x_spacer, x_pam = self.encoder(sgrnas, off_targets)
        B = len(sgrnas)
        device = x_spacer.device

        # --- NODO PAM (Lo lasciamo neurale, di solito non overfitta ed è utile per riconoscere NGG) ---
        if "pam_gate" in intervention:
            pam_gate = torch.full((B, 1), intervention["pam_gate"], device=device, dtype=torch.float32)
            _, repr_pam = self.pam_node(x_pam) 
        else:
            pam_gate, repr_pam = self.pam_node(x_pam)

        # =====================================================================
        # --- BYPASS LINEARE: CONTA MANUALE DEI MISMATCH DALLE STRINGHE ---
        # =====================================================================
        s_prox_list, s_seed_list, s_nonseed_list = [], [], []
        
        for sg, ot in zip(sgrnas, off_targets):
            # Confrontiamo nucleotide per nucleotide (1.0 = mismatch, 0.0 = match)
            # Assumiamo che i primi 20 caratteri siano lo spacer
            mismatches = [1.0 if sg[i] != ot[i] else 0.0 for i in range(20)]
            
            # DEFINIZIONE REGIONI (0-indexed, direzione 5' -> 3')
            # Allineato a models/deep/modules.py: non-seed 0:8, seed 8:16, proximal 16:20
            nonseed_mm = sum(mismatches[0:8])    # Pos 1-8 (Non-Seed / distale)
            seed_mm = sum(mismatches[8:16])      # Pos 9-16 (Seed Extension)
            prox_mm = sum(mismatches[16:20])     # Pos 17-20 (PAM-proximal)
            
            s_nonseed_list.append([nonseed_mm])
            s_seed_list.append([seed_mm])
            s_prox_list.append([prox_mm])

        # --- Nodo Proximal (Bypassato) ---
        if "proximal" in intervention:
            s_prox = torch.full((B, 1), intervention["proximal"], device=device, dtype=torch.float32)
        else:
            s_prox = torch.tensor(s_prox_list, dtype=torch.float32, device=device)
        repr_prox = torch.zeros(B, self.embed_dim, device=device) # Vettore fittizio per non rompere il return

        # --- Nodo Seed (Bypassato) ---
        if "seed" in intervention:
            s_seed = torch.full((B, 1), intervention["seed"], device=device, dtype=torch.float32)
        else:
            s_seed = torch.tensor(s_seed_list, dtype=torch.float32, device=device)
        repr_seed = torch.zeros(B, self.embed_dim, device=device)

        # --- Nodo Non-Seed (Bypassato) ---
        if "non_seed" in intervention:
            s_nonseed = torch.full((B, 1), intervention["non_seed"], device=device, dtype=torch.float32)
        else:
            s_nonseed = torch.tensor(s_nonseed_list, dtype=torch.float32, device=device)
        repr_nonseed = torch.zeros(B, self.embed_dim, device=device)

        # =====================================================================
        # --- FINE BYPASS ---
        # =====================================================================

        # --- NORMALIZZAZIONE BIOLOGICA DELLE RAPPRESENTAZIONI ---
        pam_gate = torch.sigmoid(pam_gate)
        
        # Le ReLU qui ora non faranno nulla (perché la conta è già >= 0), ma le lasciamo per sicurezza
        s_prox = F.relu(s_prox)
        s_seed = F.relu(s_seed)
        s_nonseed = F.relu(s_nonseed)

        # --- Equazione Strutturale Combinata (HARD PRIOR: SEED DOMINANCE) ---
        w_nonseed_base = F.softplus(self.w_nonseed)
        w_nonseed_eff = -w_nonseed_base
        
        w_seed_extra = F.softplus(self.w_seed)
        w_seed_eff = -(w_nonseed_base + w_seed_extra)
        
        w_prox_eff = -F.softplus(self.w_proximal)

        bias_eff = torch.clamp(self.bias, min=-4.0, max=3.0)

        logit = (s_prox * w_prox_eff) + (s_seed * w_seed_eff) + (s_nonseed * w_nonseed_eff) + bias_eff
        
        activity_prob = pam_gate * torch.sigmoid(logit)

        return {
            "pam_gate": pam_gate,
            "proximal_scalar": s_prox,
            "seed_scalar": s_seed,
            "nonseed_scalar": s_nonseed,
            "activity_probability": activity_prob,
            "repr_pam": repr_pam,
            "repr_proximal": repr_prox,
            "repr_seed": repr_seed,
            "repr_nonseed": repr_nonseed
        }
        
        # if intervention is None:
        #     intervention = {}

        # x_spacer, x_pam = self.encoder(sgrnas, off_targets)
        # B = len(sgrnas)
        # device = x_spacer.device

        # # --- Nodo PAM ---
        # if "pam_gate" in intervention:
        #     pam_gate = torch.full((B, 1), intervention["pam_gate"], device=device, dtype=torch.float32)
        #     _, repr_pam = self.pam_node(x_pam) 
        # else:
        #     pam_gate, repr_pam = self.pam_node(x_pam)

        # # --- Nodo Proximal ---
        # if "proximal" in intervention:
        #     s_prox = torch.full((B, 1), intervention["proximal"], device=device, dtype=torch.float32)
        #     _, repr_prox = self.proximal_node(x_spacer)
        # else:
        #     s_prox, repr_prox = self.proximal_node(x_spacer)

        # # --- Nodo Seed Extension ---
        # if "seed" in intervention:
        #     s_seed = torch.full((B, 1), intervention["seed"], device=device, dtype=torch.float32)
        #     _, repr_seed = self.seed_node(x_spacer)
        # else:
        #     s_seed, repr_seed = self.seed_node(x_spacer)

        # # --- Nodo Non-Seed ---
        # if "non_seed" in intervention:
        #     s_nonseed = torch.full((B, 1), intervention["non_seed"], device=device, dtype=torch.float32)
        #     _, repr_nonseed = self.nonseed_node(x_spacer)
        # else:
        #     s_nonseed, repr_nonseed = self.nonseed_node(x_spacer)

        # # --- NORMALIZZAZIONE BIOLOGICA DELLE RAPPRESENTAZIONI ---
        # # 1. Il PAM Gate DEVE essere una probabilità [0, 1] (AND logico)
        # pam_gate = torch.sigmoid(pam_gate)

        # # 2. Le penalità energetiche NON POSSONO essere negative. 
        # # Usiamo ReLU: 0 = sequenza perfetta, >0 = danno da mismatch.
        # s_prox = F.relu(s_prox)
        # s_seed = F.relu(s_seed)
        # s_nonseed = F.relu(s_nonseed)

        # # --- Equazione Strutturale Combinata (HARD CONSTRAINTS TOTALI) ---
        # # 1. Pesi termodinamici: SOLO penalità (w <= 0)
        # w_prox_eff = -F.softplus(self.w_proximal)
        # w_seed_eff = -F.softplus(self.w_seed)
        # w_nonseed_eff = -F.softplus(self.w_nonseed)

        # # 2. Bias biologico: L'attività basale non può essere < 1% o > 99.3%
        # # Usiamo clamp sul tensore del parametro per limitarne l'influenza
        # bias_eff = torch.clamp(self.bias, min=-4.0, max=3.0)

        # # 3. Logit combinato
        # logit = (s_prox * w_prox_eff) + (s_seed * w_seed_eff) + (s_nonseed * w_nonseed_eff) + bias_eff
        # activity_prob = pam_gate * torch.sigmoid(logit)

        # --- NORMALIZZAZIONE BIOLOGICA DELLE RAPPRESENTAZIONI ---
        pam_gate = torch.sigmoid(pam_gate)

        s_prox = F.relu(s_prox)
        s_seed = F.relu(s_seed)
        s_nonseed = F.relu(s_nonseed)

        # --- Equazione Strutturale Combinata (HARD PRIOR: SEED DOMINANCE) ---
        
        # 1. Il Non-Seed definisce la penalità "base" per i mismatch distali
        w_nonseed_base = F.softplus(self.w_nonseed)
        w_nonseed_eff = -w_nonseed_base
        
        # 2. Il Seed DEVE essere almeno tanto punitivo quanto il Non-Seed.
        # w_seed_extra rappresenta quanto il Seed è PIÙ importante del Non-Seed.
        # Matematicamente: |w_seed_eff| = |w_nonseed_eff| + |w_seed_extra|
        w_seed_extra = F.softplus(self.w_seed)
        w_seed_eff = -(w_nonseed_base + w_seed_extra)
        
        # 3. Proximal rimane indipendente (solitamente tra Seed e Non-Seed come importanza)
        w_prox_eff = -F.softplus(self.w_proximal)

        # 4. Bias biologico: clamp per evitare l'esplosione dei logit
        bias_eff = torch.clamp(self.bias, min=-4.0, max=3.0)

        # 5. Logit combinato: ora la gerarchia è garantita dall'algebra
        logit = (s_prox * w_prox_eff) + (s_seed * w_seed_eff) + (s_nonseed * w_nonseed_eff) + bias_eff
        
        activity_prob = pam_gate * torch.sigmoid(logit)

        return {
            "pam_gate": pam_gate,
            "proximal_scalar": s_prox,
            "seed_scalar": s_seed,
            "nonseed_scalar": s_nonseed,
            "activity_probability": activity_prob,
            "repr_pam": repr_pam,
            "repr_proximal": repr_prox,
            "repr_seed": repr_seed,
            "repr_nonseed": repr_nonseed
        }

    def forward(self, sgrnas: list[str] | str, off_targets: list[str] | str) -> dict[str, torch.Tensor]:
        """Esecuzione standard osservazionale."""
        if isinstance(sgrnas, str): 
            sgrnas = [sgrnas]
        if isinstance(off_targets, str): 
            off_targets = [off_targets]
        return self._base_forward(sgrnas, off_targets)

    def do(self, sgrnas: list[str] | str, off_targets: list[str] | str, intervention: dict[str, float]) -> dict[str, torch.Tensor]:
        """Esecuzione sotto intervento causale (G-computation forward)."""
        if isinstance(sgrnas, str): 
            sgrnas = [sgrnas]
        if isinstance(off_targets, str): 
            off_targets = [off_targets]
        return self._base_forward(sgrnas, off_targets, intervention=intervention)

    def predict_proba_batch(self, sgrnas: list[str], off_targets: list[str]) -> torch.Tensor:
        """Restituisce Tensor[B] di probabilità — per training e valutazione."""
        out = self._base_forward(sgrnas, off_targets)
        return out["activity_probability"].squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, sgrna: str, off_seq: str) -> float:
        """Helper per inferenza singola (scollegato dal grafo computazionale)."""
        out = self.forward([sgrna], [off_seq])
        return float(out["activity_probability"].item())

    @torch.no_grad()
    def explain(self, sgrna: str, off_seq: str) -> dict[str, float]:
        """Restituisce il contributo esatto di ogni sottomodulo causale."""
        out = self.forward([sgrna], [off_seq])
        return {
            "pam_gate": float(out["pam_gate"].item()),
            "proximal_penalty": float(out["proximal_scalar"].item()),
            "seed_penalty": float(out["seed_scalar"].item()),
            "nonseed_penalty": float(out["nonseed_scalar"].item()),
            "final_probability": float(out["activity_probability"].item())
        }