# Findings

Documento di ricerca per tracciare osservazioni empiriche, anomalie e decisioni metodologiche emerse durante gli esperimenti. Aggiornare ad ogni fase.

---

## Fase 1 — Baseline XGBoost/CatBoost con feature DAG

### F1 — Generalizzazione cross-assay: calo netto di AUPRC con AUROC alto

**Esperimento:** valutazione within-dataset (CHANGE-seq) e cross-assay (GUIDE-seq).

**Risultato:**

| Modello | AUPRC within | AUPRC cross | Delta | AUROC cross |
|---|---|---|---|---|
| XGBoost | 0.4393 | 0.1853 | -57% | 0.966 |
| CatBoost | 0.4916 | 0.2744 | -44% | 0.974 |

**Interpretazione:** il modello sa ordinare correttamente i positivi su GUIDE-seq (AUROC alto) ma è mal calibrato — il threshold ottimizzato su CHANGE-seq non trasferisce. Il problema non è la rappresentazione ma la distribuzione degli score tra i due assay. Questo è un segnale di overfitting al protocollo sperimentale di CHANGE-seq, non al meccanismo biologico sottostante.

**Implicazione per la tesi:** motiva l'uso di un modello con vincoli causali strutturali che apprendano il meccanismo e non le correlazioni specifiche dell'assay.

---

### F2 — Feature energetiche aggregate peggiorano la performance

**Esperimento:** ablation study con varianti `no_aggregate_energy`, `no_energy_full`, `full_dag`.

**Risultato:**

| Variante | XGBoost AUPRC | CatBoost AUPRC |
|---|---|---|
| no_energy_full | 0.4584 | 0.5310 |
| no_aggregate_energy | 0.4489 | 0.5332 |
| full_dag | 0.4393 | 0.4916 |

**Interpretazione:** le feature energetiche aggregate (`mean_energy_penalty`, `total_energy_penalty`, `concept_energy`) peggiorano la performance rispetto alla loro rimozione. I nodi energetici nodali (`node_B_proximal`, `node_C_seed_extension`, `node_D_non_seed`) non aggiungono informazione significativa rispetto alle feature di conteggio — la differenza tra `no_aggregate_energy` e `no_energy_full` è < 0.01 per entrambi i modelli.

**Ipotesi:** i pesi energetici (wobble=0.4, transition=0.75, transversion=1.0) non sono calibrati ottimalmente sui dati CHANGE-seq. Aggiungono rumore perché sono collineari con `mismatch_count` ma con una pesatura arbitraria.

**Decisione:** usare `no_aggregate_energy` come configurazione base per tutti gli esperimenti successivi. I pesi energetici andranno stimati dai dati nell'SCM in fase 2 invece di essere assunti a priori.

---

### F3 — PAM solo non è informativo come feature

**Esperimento:** ablation `pam_only`.

**Risultato:** AUPRC = 0.04 (quasi casuale) per entrambi i modelli.

**Interpretazione:** quasi tutti i target nel dataset CHANGE-seq hanno PAM canonico NGG — il PAM da solo non discrimina perché non c'è variabilità sufficiente. Il PAM è invece un gate causale forte (biologicamente il primo checkpoint del meccanismo Cas9), ma la sua rilevanza emerge solo in interazione con le altre feature, non marginalmente.

**Implicazione per il DAG:** il PAM va modellato come gate moltiplicativo nell'SCM (fase 2), non come feature additiva come fa attualmente XGBoost.

---

### F4 — node_D_non_seed fallisce il test esterno: confounding nel DAG

**Esperimento:** validazione DAG, test esterno `node_D_non_seed → label`.

**Risultato:**
```
node_D_non_seed → label:  ρ = +0.028  atteso NEGATIVO  ✗ FAIL
```

**Interpretazione:** la correlazione marginale di `node_D` con la label è positiva, contrariamente all'ipotesi causale (più energia in non-seed = meno attività). Il motivo è confounding strutturale: i target con alta energia in non-seed ma bassa in seed tendono ad essere off-target attivi perché la seed è intatta. L'effetto di `node_D` non è diretto ma mediato e condizionato a `node_B` e `node_C`.

**Revisione DAG da valutare (da testare con independence tests in fase 2):**

- *Opzione A:* rimuovere l'arco diretto `node_D → activity` e modellare `node_D` come modificatore di `full_hybridization`:
  ```
  Prima:  node_D → activity  (arco diretto)
  Dopo:   node_D → full_hybridization → activity  (mediato)
  ```
- *Opzione B:* aggiungere un arco di interazione `node_B × node_D` — l'effetto di `node_D` è negativo solo quando `node_B` è basso (seed intatta).

**Da fare in fase 2:** testare con `dag/independence_tests.py` quale delle due opzioni è supportata dalle indipendenze condizionali nei dati.

---

### F5 — Correlazioni esterne del DAG molto basse ma nella direzione attesa

**Esperimento:** validazione DAG, test esterni verso `label`.

**Risultato:**
```
node_A_pam            → label:  ρ = +0.044  ✓
node_B_proximal       → label:  ρ = -0.038  ✓
node_C_seed_extension → label:  ρ = -0.049  ✓
mismatch_count        → label:  ρ = -0.040  ✓
```

**Interpretazione:** le correlazioni sono nella direzione biologicamente attesa ma hanno magnitudine molto bassa. Questo è atteso: le relazioni sono non lineari, il dataset è fortemente sbilanciato (41x), e le correlazioni marginali di Spearman sottostimano le relazioni condizionali. Non invalida il DAG — indica che le relazioni causali emergono solo condizionando sugli altri nodi, non marginalmente.

---

## Todo — Da investigare nelle fasi successive

- [ ] **Fase 2:** testare indipendenza condizionale `node_D ⊥ activity | node_B, node_C` per decidere la revisione del DAG (F4)
- [ ] **Fase 2:** stimare i pesi energetici α, β, γ dai dati invece di assumerli a priori (F2)
- [ ] **Fase 2:** implementare PAM come gate moltiplicativo nell'SCM e misurare impatto su CCS (F3)
- [ ] **Fase 2:** calcolare CCS sul baseline XGBoost come punto di riferimento
- [ ] **Fase 3:** verificare se il Neural SCM risolve il calo cross-assay (F1)
