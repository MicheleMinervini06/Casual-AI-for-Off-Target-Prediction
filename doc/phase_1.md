# Fase 1 — Cosa manca
Tre fix prima di chiudere la fase:
* dag/validate.py — cambia il target da mismatch_rate a label in DAG_EDGES_TO_VALIDATE. La validazione attuale non testa causalità verso l'outcome.
* experiments/configs/base.yaml — rimuovi le feature energetiche da feature_cols. L'ablation ha dimostrato che no_energy batte full_dag.
* evaluation/ccs.py — implementa il Causal Consistency Score. Serve come ponte tra fase 1 e fase 2: misura già ora quanto il modello rispetta i vincoli causali, prima di avere l'SCM.