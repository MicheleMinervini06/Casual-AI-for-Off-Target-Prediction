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

### Aggiornamento — Risultati con nuovo split (exp_01)

Analisi risultati exp_01 con nuovo split

Metriche predittive
- XGBoost:  AUPRC within=0.144  cross=0.270
- CatBoost: AUPRC within=0.183  cross=0.231

AUPRC within è sceso rispetto al run precedente (0.44/0.49) — atteso. Il test set ora è più rappresentativo e bilanciato. Il dato interessante è che cross-assay è migliorato per XGBoost (0.185→0.270) — il modello generalizza meglio con split corretto.

DAG Validation — nuovo finding
- node_C_seed_extension → label:  ρ=+0.011  atteso NEGATIVO  ✗ FAIL
- node_D_non_seed       → label:  ρ=+0.071  atteso NEGATIVO  ✗ FAIL

Con il nuovo split ora falliscono sia node_C che node_D — prima falliva solo node_D. Con il vecchio split le guide ricche di positivi nel val mascheravano questo pattern.

Ablation — confermato
- CatBoost no_energy_full:      0.2055  ← migliore
- CatBoost no_aggregate_energy: 0.2019
- CatBoost full_dag:            0.1829

Il pattern precedente regge: le feature energetiche aggregate peggiorano la performance.

Feature importance — segnale chiave
- mismatch_count:  0.134  ← feature più importante
- pam_score:       0.109
- profile_pos_20:  0.076  ← posizione PAM-proximal
- gc_sgRNA:        0.069
- node_B_proximal: 0.015  ← nodo DAG poco usato
- node_C_seed:     0.015

XGBoost usa principalmente feature aggregate (mismatch_count, mismatch_rate) e il profilo posizionale, non i nodi DAG strutturati. Questo è coerente con R2=0.0 nel CCS — il modello non ha imparato l'effetto causale posizione-specifico.

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

**Aggiornamento dopo Fase 2:** l'ipotesi di confounding biologico resta plausibile, ma non e' sufficiente a spiegare tutte le dipendenze anomale osservate nei test CI globali. Parte del segnale e' verosimilmente dovuto a bias di costruzione del dataset (vedi F6.2).

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

## Fase 2 — SCM parametrico + do-calculus + independence tests

### F6.1 — Test di indipendenza condizionale: alta violazione delle implicazioni del DAG

**Esperimento:** `exp_02_scm` con DAG potato (`node_A_pam`, `node_B_proximal`, `node_C_seed_extension`, `pam_score`, `mismatch_rate`, `label`) e test CI multipli con correzione FWER.

**Risultato:**

- `ci_failure_rate = 0.8889` (8/9 test respingono H0)
- esempio coerente col DAG: `label ⟂ node_A_pam | pam_score` con `p = 0.703`
- molte altre indipendenze teoriche non reggono sui dati osservati

**Interpretazione:** il DAG semplificato non descrive completamente la struttura statistica osservata. Tuttavia, il risultato non va letto automaticamente come "DAG biologicamente sbagliato": la pipeline negativa introduce dipendenze spurie (vedi F6.2).

---

### F6.2 — Dipendenza PAM-mismatch: probabile artefatto di costruzione dei negativi (Cas-OFFinder)

**Osservazione chiave:** nei test CI emerge una dipendenza forte tra variabili derivate da porzioni diverse della sequenza (PAM vs mismatch rate/profili mismatch), non attesa dal meccanismo causale biologico puro.

**Ipotesi supportata:** i negativi generati con Cas-OFFinder (fino a 6 mismatch, campionati con regole non condizionate sul PAM biologico) introducono una correlazione artificiosa tra PAM e mismatch nei negativi. Quindi una quota rilevante delle violazioni CI e' un **dataset bias**, non una misspecification meccanicistica del DAG.

**Implicazione metodologica:** separare la valutazione in:

- coerenza biologica del DAG/SCM
- robustezza rispetto al processo di negative sampling

e usare test stratificati per assay/source o reweighting dei negativi prima di concludere sulla struttura causale.

---

### F6.3 — SCM parametrico: parametri con segni biologicamente plausibili

**Risultato (fit train):**

- `pam_alpha = 1.00`
- `activity_delta_pam = +4.46`
- `activity_eta_proximal = -6.83`
- `activity_theta_seed = -13.33`

**Interpretazione:** il gate PAM e il costo mismatch (soprattutto seed-extension) sono coerenti con il meccanismo atteso. Il problema principale non e' il segno dei coefficienti, ma il mismatch tra modello causale ideale e distribuzione empirica dei campioni (bias + possibile eterogeneita' assay).

---

### F6.4 — CCS baseline (3 regole): coerenza causale ancora insufficiente

**Esperimento:** CCS sul baseline XGBoost con `mode = 3_rules`.

**Risultato:**

- `R1_PAM_Ablation = 1.0`
- `R2_Pos1_Mismatch = 0.0`
- `R3_Heal_Seed = 0.9375`
- `CCS_Overall = 0.0`

**Interpretazione:** anche con metrica ridotta (3 regole) il baseline fallisce la coerenza causale complessiva per violazione netta di R2. Quindi il passaggio a 3 regole non cambia la conclusione qualitativa: il baseline non e' causalmente consistente in modo globale.

---

### F6.5 — Dataset interventivo: fissata inconsistenza interna

**Problema precedente:** `build_intervention_dataset` applicava i valori di intervento senza propagare le variabili a valle, producendo blocchi con feature intervenute ma label non aggiornate.

**Fix implementato:** propagazione opzionale tramite SCM (`scm` opzionale) durante la costruzione del dataset sintetico.

**Esito:** il dataset interventivo ora include `activity_probability` coerente con l'intervento; `label` e' allineata alla soglia su probabilita' nel blocco sintetico.

---

## Todo — Da investigare nelle fasi successive

- [x] **Fase 2:** testare indipendenza condizionale con `dag/independence_tests.py` (eseguito; outcome fortemente influenzato da bias dei negativi da validare con analisi stratificate)
- [x] **Fase 2:** stimare i pesi energetici/strutturali dai dati nell'SCM parametrico (versione potato)
- [x] **Fase 2:** implementare PAM come gate moltiplicativo nell'SCM
- [x] **Fase 2:** calcolare CCS sul baseline XGBoost come punto di riferimento
- [ ] **Fase 2-bis:** quantificare esplicitamente il bias Cas-OFFinder (analisi separata positivi/negativi e per assay)
- [ ] **Fase 2-bis:** ripetere CI tests con campionamento negativo controllato o reweighting
- [ ] **Fase 3:** verificare se il Neural SCM risolve il calo cross-assay (F1)

### F6.6 — Confronto: default DAG vs variante con arco mismatch_rate → label

La modifica del DAG che aggiunge l'arco diretto `mismatch_rate → label` non porta benefici rilevanti sugli esiti sperimentali. I segnali principali sono i seguenti:

- Il CI failure rate scende da 0.889 a 0.875 — un miglioramento di 1.4 punti percentuali su un failure rate dell'87%. Non è statisticamente o praticament rilevante.
- Il delta osservazionale vs interventale cambia di circa 0.0003 — compatibile con rumore statistico.
- L'aspetto più notevole è che il fit per la variante assegna `activity_eta_mismatch ≈ -30`, ovvero un peso estremamente grande a `mismatch_rate`.

Questa grandezza dei coefficienti è coerente con un problema di collinearità: `mismatch_rate` è una funzione di `node_B_proximal` e `node_C_seed_extension`, quindi l'aggiunta dell'arco introduce ridondanza informativa che il fitting compensa con coefficienti estremi invece di migliorare la rappresentazione causalmente.

Conclusione: la variante non migliora la qualità causale o predittiva degli esperimenti. Il DAG originale (senza l'arco diretto `mismatch_rate → label`) è preferibile per parsimonia: cattura l'informazione necessaria tramite i nodi esistenti e evita ridondanze che portano a stime instabili. Non portare questa modifica in Fase 3; il risultato è comunque un finding utile che conferma come i segnali evidenziati dai test di indipendenza siano riconducibili a informazioni ridondanti nei nodi già presenti, non alla necessità di un arco esplicito.

---

## Fase 3 — Neural SCM: deep complesso vs bypass lineare

### F7 — La complessità deep induce overfitting severo; il bypass lineare semplificato migliora la generalizzazione

**Esperimento:** `Exp04_LinearBypass_HardPrior` (encoder biologico + bypass lineare delle regioni mismatch, training con Focal Loss, OneCycleLR, 5 epoche).

**Setup rilevante:**

- split: train=2,925,972 / val=842,659 / test=582,999
- imbalance train: `pos_weight = 46.3`
- device: CUDA
- encoder: `BiologicalMismatchEncoder (embed_dim=12)`
- combinatore strutturale con pochi gradi di libertà (assetto semplificato):
  - `w_proximal`, `w_seed`, `w_nonseed`, `bias` + gate PAM

**Risultati Exp04:**

| Split | AUPRC | AUROC | F1 |
|---|---:|---:|---:|
| CHANGE-seq train | 0.2362 | 0.8834 | 0.0005 |
| CHANGE-seq val | 0.0469 | 0.7853 | 0.0014 |
| CHANGE-seq test | 0.0878 | 0.7982 | 0.0009 |
| GUIDE-seq cross-assay | 0.1250 | 0.8722 | 0.0094 |

**Causal Consistency:**

- `Neural CCS_Overall = 0.8333` (decisamente migliore rispetto ai baseline non causali)

**Confronto con i run deep precedenti (prima del bypass lineare):**

- nei run con maggiore componente deep si osservava pattern di overfitting severo, con train AUPRC molto alto e validation quasi collassata (ordine di grandezza ~`0.89` train vs ~`0.002` val)
- con il modello semplificato il gap train/val resta presente ma si riduce in modo sostanziale; la validazione torna su valori non degeneri (`~0.047`)

**Interpretazione:**

- la porzione deep ad alta capacità nella catena causale tendeva a catturare correlazioni spurie assay-specifiche
- il ritorno a una forma più parsimoniosa (bypass lineare con hard priors) riduce la varianza del modello e migliora la robustezza out-of-sample
- il miglioramento di CCS suggerisce che la semplificazione non solo aiuta la generalizzazione predittiva, ma preserva meglio la coerenza causale desiderata

**Implicazione per la tesi:**

- in questo dominio, una parametrizzazione causale semplice e ben vincolata è preferibile a una componente deep più espressiva ma instabile
- la complessità architetturale va introdotta solo se produce un guadagno netto e stabile su validation/cross-assay, non solo su train

---

## Fase 4 — Analisi controfattuale batch su Neural SCM (Exp15_Positional)

Estensione di `simulate_intervention.py` da single-pair a batch su `CHANGEseq_positive.csv` (67k coppie, 104 guide) e `GUIDEseq_positive.csv` (1616 coppie, 46 guide). Pipeline Pearl classica: abduzione `U = y_obs_logit - y_pred_logit`, intervento `do(...)`, predizione controfattuale `y_cf = sigmoid(y_do_logit + U)`. Interventi fissi: truncation 5' (`NN + guide[2:]`) e mutation pos15→A (`guide[14] = 'A'`).

### F8 — La saturazione di `y_obs_on` rendeva l'abduzione on-target degenere

**Problema identificato:** `reads_to_prob(on_reads, on_reads, "log") ≡ 99%` per costruzione (numeratore = denominatore, capped a 99). Quindi `U_on = logit(0.99) - y_pred_on_logit` ha correlazione **−0.994** con `y_pred_on_prob`: completamente determinato dal modello, non un'inferenza dai dati. Non è abduzione, è uno shift di calibrazione.

**Conseguenza pre-fix:** su single-pair, `99% → 96.9%` post-intervento sembrava un trade-off accettabile. In realtà U_on (~+2.6 logit) agiva da ammortizzatore costante che mascherava qualunque crollo di `logit_on_intervened`.

**Fix implementato:** due regimi distinti, selezionabili via `--on-target-mode`:

- `per_run` (default GUIDEseq): denominatore di `reads_to_prob` = max reads del `run` sperimentale. `y_obs_on_prob` varia per guida, U_on diventa una vera abduzione (mean=−0.55, std=1.38).
- `drop` (default CHANGEseq, manca colonna `run`): nessuna abduzione on-target. Baseline = `y_pred_on_prob`. `delta_on` = credenza del modello sull'effetto dell'intervento. Asimmetrico rispetto a `delta_off` ma epistemicamente onesto.

**Implicazione:** la saturazione era un bug interpretativo, non di magnitudine. I risultati pre-fix sottostimavano sistematicamente il costo on-target degli interventi (truncation 5' passa da `Δon = -1.3%` pre-fix a `Δon = -15.5%` post-fix su GUIDEseq). Tutte le conclusioni single-pair dello script originale vanno riviste sotto il nuovo regime.

---

### F9 — `U_off` cross-assay: scaling cell-free vs in vivo, non errore di modello

**Osservazione:** post-fix, `U_off` ha distribuzioni molto diverse tra training set (CHANGEseq, in vitro) e test set (GUIDEseq, in vivo).

| Dataset | U_off mean | U_off std | forma |
|---|---:|---:|---|
| CHANGEseq (train) | **+2.34** | 1.36 | bimodale (~+1, ~+4) |
| GUIDEseq (test) | **−0.14** | 1.22 | unimodale, ~normale |

**Diagnostica chiave** — rapporto raw `off_reads / on_reads`:

| | mediana | q95 | max |
|---|---:|---:|---:|
| CHANGEseq | **0.667** | 2.23 | **30.8** |
| GUIDEseq | **0.009** | 0.52 | 4.98 |

In vitro l'off-target medio ha 2/3 dell'attività on-target e alcuni la superano di 30×; in vivo lo 0.9%. Differenza di due ordini di grandezza nella scala operativa.

**Predizioni del modello — invariate** tra dataset (CHANGEseq median 67%, GUIDEseq median 54%): il modello ha imparato la termodinamica guida–DNA che è invariante al regime sperimentale. È `y_obs_off_prob` a cambiare scala (CHANGEseq median 93% vs GUIDEseq 42%) perché `reads_to_prob` riflette l'output sperimentale grezzo.

**Verifica empirica** — stratificazione di `U_off` per `distance` (numero di mismatch):

| distance | U_off CHANGEseq | U_off GUIDEseq |
|---:|---:|---:|
| 1 | +1.94 | +0.91 |
| 4 | +2.10 | −0.29 |
| 6 | **+2.67** | +0.03 |

Su CHANGEseq il gap cresce monotonicamente con i mismatch: il modello sa che 6 mismatch dovrebbero ridurre l'attività, ma in vitro questi siti restano saturi vicino all'on-target. Su GUIDEseq il pattern è piatto, coerente con la termodinamica.

**Bimodalità su CHANGEseq:**

- mode ~+1: off-target con attività intermedia (rapporto reads 0.3–0.7)
- mode ~+4: off-target con `off_reads ≥ on_reads` (rapporto > 1, fino a 30) — saturazione tecnica del cell-free

**Interpretazione:** `U_off` su CHANGEseq non misura errore casuale del modello, misura la **distanza tra termodinamica pura e biofisica del cell-free**. Il modello non sbaglia: è la metrica `y_obs` ad avere una scala diversa nei due regimi.

**Implicazione metodologica:** `delta = y_cf - y_obs` è invariante a uno shift sistematico di U (si compensa nel cambio), quindi i controfattuali restano validi su entrambi i dataset. Cambia l'interpretazione semantica:

- GUIDEseq: `delta_off` = cambiamento atteso nell'attività in vivo osservata
- CHANGEseq: `delta_off` = cambiamento in scala "termodinamica" proiettato via U sulla scala in-vitro

GUIDEseq resta il regime epistemicamente affidabile per claim clinici. CHANGEseq è utile come test della termodinamica del modello e come diagnostica del gap in-vitro/in-vivo. Coerente con F1: il modello generalizza il **meccanismo**, non il **protocollo**.

---

### F10 — Interventi fissi non superano il test del Pareto su nessun dataset

**Metrica:** quadrante ideale = `(Δoff < 0) ∧ (Δon ≥ −5%)`. Aggregazione per-guida (mediana entro guida → distribuzione su N guide).

**Risultato:**

| Dataset | N guide | Truncation 5' ideale | Mutation pos15→A ideale |
|---|---:|---:|---:|
| GUIDEseq (per_run) | 46 | 3 (6.5%) | 5 (10.9%) |
| CHANGEseq (drop) | 104 | **0 (0.0%)** | **0 (0.0%)** |

**Magnitudine media post-fix (GUIDEseq, per coppia):**

- Truncation 5': `Δoff = −12.9%`, `Δon = −15.5%` → distrugge entrambi quasi simmetricamente
- Mutation pos15→A: `Δoff = −7.95%`, `Δon = −7.17%` → trade-off neutro in media

**Patologia della Mutation pos15→A:** `q25(Δoff_mut) = 0.00` esatto perché ~25% delle guide hanno già `A` in posizione 14 (no-op). Inoltre **43.4% delle coppie ha `Δoff ≥ 0`**: la mutazione spesso porta la guida più vicina all'off-target invece che più lontana, perché la pos 15 non è scelta in modo guida-specifico.

**Interpretazione:** il single-pair output positivo (`99% → 22.9%` off-target) era artefatto della saturazione di U_on (F8). Una volta calibrata l'abduzione, gli interventi fissi non offrono trade-off accettabile su scala batch.

**Implicazione:** la prossima iterazione richiede rescue mutation **guida-specifica**: per ogni coppia, scegliere posizioni `i ∈ seed-extension` dove `on_target[i] ≠ off_target[i]` e mutare `guide[i] = on_target[i]`. Solo così l'intervento separa on-target e off-target invece di degradare entrambi simmetricamente.

---

### F11 — Aggregazione per-guida vs per-coppia: differenze sostanziali nelle conclusioni

**Problema:** GUIDEseq ha guide con 543 righe e altre con 1 riga; CHANGEseq stesso pattern. Le statistiche su tutta la popolazione di coppie sono dominate dalle poche guide con molti off-target rilevati.

**Esempio quantitativo (GUIDEseq, Mutation pos15→A):**

- mean `Δoff` su tutte le coppie (N=1616): `−7.95%`
- mean delle mediane per-guida (N=46): `−13.5%`

Differenza ~6 punti percentuali: una guida con 543 righe pesa 543× nella media globale. Entrambi i numeri sono "veri", ma rispondono a domande diverse.

**Decisione metodologica:**

- per claim del tipo "efficacia attesa di un intervento su una guida nuova": **mediana entro guida → distribuzione di quei valori** (per-guida)
- per claim del tipo "copertura sui dati osservati nel dataset": media globale (per-coppia)

**Implementazione:** lo script salva entrambi gli output (`<dataset>_batch_results.csv` per-coppia + `<dataset>_per_guide_medians.csv` per-guida). I quadranti ideali nel sommario riportano entrambe le aggregazioni così le conclusioni sono leggibili senza ambiguità.

---

## Todo Fase 4

- [ ] Implementare rescue mutation guida-specifica (sostituisce `mutate_pos15` fissa). Per ogni coppia: posizioni `i ∈ [8, 15]` dove `on_target[i] ≠ off_target[i]`, mutare `guide[i] = on_target[i]`. Filtrare coppie senza posizioni qualificanti.
- [ ] Stratificare `U_off` su CHANGEseq per GC%, regione genomica accessibile, distanza al PAM canonico per testare l'ipotesi "U_off = saturazione cell-free" (F9).
- [ ] Aggiungere bootstrap CI sulle medie per-guida dei `Δ` per quantificare la significatività statistica delle conclusioni F10.
- [ ] Confrontare interventi fissi vs rescue guida-specifica sulla stessa popolazione per validare il ragionamento controfattuale del modello.

---

## Fase 5 — Estensione variazionale: modellazione esplicita di U con ELBO

### Motivazione

Nel framework di Pearl ogni nodo del DAG riceve un termine esogeno `U_i` che cattura tutto ciò che il modello strutturale non spiega. Fino a Run 15, U non era modellato esplicitamente: il backbone calcolava `P(Y|X)` con `U=0` (media del prior), che è teoricamente corretto per la predizione osservazionale ma non fornisce una distribuzione posteriore `q(U|X, Y_obs)` per l'abduction individuale.

L'obiettivo era introdurre un encoder `q(U|X, Y)` addestrato con l'ELBO (Evidence Lower Bound):

```
ELBO = E_q[log p(Y | X, U)] − KL[q(U|X,Y) || p(U)]
```

con `p(U) = N(0,1)` e U iniettato additivamente nel logit finale (`logit_cf = struct_logit + U`). Il termine KL regolarizza l'encoder verso il prior, prevenendo che U diventi un semplice lookup della label.

---

### F12 — Run 16 (β=1.0): KL collapse — encoder rate-limited dall'information bottleneck

**Setup:** encoder `q(U|X,Y)` con input 2-dim `(structural_logit, y)`, β=1.0, warmup KL 5 epoche.

**Nota architetturale:** la prima versione dell'encoder usava 81 feature come input (80 feature di sequenza + 1 label). Il problema era che le feature di sequenza dominavano il singolo bit di label, impedendo all'encoder di imparare l'abduction del residuo. Fix: ridotto l'input a 2 dimensioni `(structural_logit, y)`, forzando l'MLP a modellare la discrepanza tra predizione strutturale e osservazione — semanticamente equivalente al residuo di abduction di Pearl.

**Diagnostica (val split, 842k campioni):**

| Metrica | Valore | Interpretazione |
|---|---|---|
| mean KL | 6.8×10⁻⁵ | quasi zero → collapse |
| μ_U std | 0.011 | encoder produce μ ≈ 0 per quasi tutti gli esempi |
| pearson(μ, label) | 0.81 | encoder direzionalmente corretto |
| decoder_sensitivity | 3.4×10⁻⁴ | backbone quasi ignora U |
| verdict | full_collapse | — |

**Spiegazione teorica:** si può dimostrare (Higgins et al., β-VAE, 2017; Alemi et al., 2018) che:

```
I(U ; Y | X)  ≤  KL[ q(U|X,Y) || p(U) ]
```

La mutua informazione tra U e Y dato X è limitata superiormente dalla KL. Con β=1.0 la penalità KL sopprime qualunque deviazione dal prior → l'encoder non riesce ad amplificare il segnale già appreso direzionalmente. Non è un collapse per assenza di segnale, ma un collapse per pressione eccessiva del regolarizzatore.

**Risultati predittivi:**

| Split | AUPRC | AUROC |
|---|---|---|
| CHANGE-seq test | 0.139 | 0.896 |
| GUIDE-seq cross-assay | 0.209 | 0.950 |

Leggermente inferiore a Run 15 (0.154 / 0.285): il KL aggiunge rumore al training senza portare beneficio perché U ≈ 0 sempre.

---

### F13 — Run 17 (β=0.1, β-VAE): label leakage — backbone lazy

**Setup:** identico a Run 16 con `beta_kl_max = 0.1` (ridotto di 10×). Logica: abbassare β rilascia il bottleneck di ampiezza permettendo all'encoder di amplificare il segnale già direzionalmente corretto.

**Diagnostica (val split):**

| Metrica | Run 16 (β=1.0) | Run 17 (β=0.1) | Trend |
|---|---|---|---|
| mean KL | 6.8×10⁻⁵ | 4.0×10⁻³ | ×60 ↑ |
| μ_U std | 0.011 | 0.081 | ×8 ↑ |
| pearson(μ, label) | 0.81 | 0.85 | stabile ↑ |
| decoder_sensitivity | 3.4×10⁻⁴ | 2.6×10⁻³ | ×8 ↑ |

L'encoder è diventato attivo. Ma i risultati predittivi sono crollati:

| Split | Run 15 (ref) | Run 17 | Delta |
|---|---|---|---|
| CHANGE-seq test AUPRC | 0.154 | 0.044 | −71% |
| GUIDE-seq AUPRC | 0.285 | 0.021 | −93% |

**Diagnosi — Label leakage con lazy backbone:**

Con β=0.1 il training minimizza `Focal(struct + U_sample, Y)` dove `U_sample = μ + σε` e `μ = encoder(struct_logit, Y)`. L'encoder ha accesso diretto a `Y` → apprende `μ ≈ f(Y)`. Il gradiente della loss fluisce principalmente attraverso U, quindi il backbone apprende a fare poco e delegare a U:

```
training:   U = encoder(struct_logit, Y)  → informativo → backbone lazy
inferenza:  U = 0  (Y non disponibile)    → backbone debole → crash
```

Questa è una tensione strutturale del CVAE end-to-end: l'encoder è un "shortcut" che impedisce al backbone di imparare il meccanismo strutturale.

---

### F14 — Fix A (MC marginalization): non risolve il lazy backbone

**Fix proposto:** all'inferenza, invece di `U=0`, calcolare `p(Y|X) = E_{U~N(0,1)}[p(Y|X,U)]` con K=16 campioni dal prior.

**Risultato:**

| Split | Run 17 (U=0) | Run 17 + Fix A (MC, K=16) |
|---|---|---|
| CHANGE-seq test AUPRC | 0.044 | 0.0435 |
| GUIDE-seq AUPRC | 0.021 | 0.021 |

Nessun miglioramento. La MC marginalization funziona solo se il backbone è già predittivo: `E[σ(struct + U)] ≈ σ(struct / √(1 + πσ²/8))` — se `struct ≈ 0` (backbone lazy), l'attesa rimane ≈ 0.5 per ogni esempio e la discriminazione è nulla.

---

### F15 — Conclusione: l'abduction algebrica post-hoc è la soluzione corretta e già implementata

**Analisi teorica:** nella teoria classica di Pearl, U non viene mai ottimizzato congiuntamente al modello strutturale. Le equazioni strutturali `f_i` vengono apprese prima (dalla distribuzione osservazionale o interventionale), dopodiché U viene inferito come residuo:

```
U* = logit(Y_obs) − struct_logit(X)   ← abduction algebrica (forma chiusa)
Y_cf = σ(struct_logit(X_cf) + U*)     ← controfattuale individuale
```

Questa forma è identica a ciò che `simulate_intervention.py` e `simulate_intervention_batch.py` già implementano sul backbone di Run 15, senza alcun training aggiuntivo.

**Perché l'end-to-end ELBO è inferiore per questo problema:**

1. Per la predizione osservazionale `P(Y|X)`: U si marginalizza sul prior → equivalente a `U=0` → backbone puro è sufficiente e corretto (Run 15)
2. Per i controfattuali individuali: serve `U* = abduction(Y_obs)` → la forma algebrica è più fedele a Pearl, più semplice, e non introduce il lazy backbone problem
3. Il CVAE end-to-end avrebbe senso solo se il backbone fosse vincolato a restare predittivo anche con `U=0` (backbone supervision) — ma questo richiederebbe un training a due fasi, riproducendo esattamente la separazione teorica di Pearl

**Configurazione finale adottata:**

| Componente | Implementazione | Note |
|---|---|---|
| Modello predittivo `P(Y\|X)` | Run 15 backbone (positional MLP, no U) | Pearl-consistent per predizione osservazionale |
| CCS — effetti causali medi | `model.do()` su nodi DAG | nessun U necessario per ATE |
| Abduction individuale `U*` | `U* = logit(y_obs) − struct_logit(x)` | algebraica, post-hoc, zero overhead di training |
| Controfattuale individuale | `σ(struct_logit(x_cf) + U*)` | già operativo in `simulate_intervention_batch.py` |

**Nota sulla pam_gate:** l'abduction algebrica ignora il fattore moltiplicativo `pam_gate` (`activity_prob = pam_gate × σ(logit)`). La forma esatta non è risolvibile in forma chiusa con pam_gate. Per off-target con PAM canonico (NGG, `pam_gate ≈ 1`) l'approssimazione è trascurabile; per PAM non canonici introduce un bias sistematico. Documentato come limitazione (vedi F17 per l'evidenza empirica del problema reale).

---

## Fase 6 — Limiti architetturali di positional_mlp: evidenza empirica

Dopo aver corretto l'abduction in formula `pam_gate`-aware (F15) e aver aggiunto due nuovi interventi mirati a sondare i limiti rappresentazionali del modello, sono emerse evidenze empiriche dirette di limiti che fino a Run 15 erano solo ipotizzati.

### F16 — Conferma empirica della blindness inter-posizionale

**Esperimento:** estensione di `simulate_intervention_batch.py` con due nuovi interventi a livello sequenza, eseguiti sull'intero dataset GUIDEseq positivo (1616 coppie, 46 guide):

1. **Diversity intervention** — Treatment vs Control sulle 4 posizioni PAM-prossimali (indici 16-19):
   - Treatment T: `guide[16:20] = "ACGT"` (massima diversità A/C/G/T)
   - Control C: `guide[16:20] = "AAAA"` (nessuna diversità)
2. **Repeat intervention** — Treatment vs Control sul seed (indici 8-15):
   - Treatment T: `guide[8:16] = "ATATATAT"` (perfect period-2 repeat)
   - Control C: `guide[8:16] = "AAAATTTT"` (stessa composizione A/T, no period-2)

Per ciascuno, calcolato il contrasto `delta_TC = y_cf_T − y_cf_C` come metrica dell'effetto "puro" dell'intervento isolato dal contesto della coppia.

**Risultati (per coppia):**

| Intervento | `delta_T` off | `delta_C` off | **contrast T-C** off |
|---|---:|---:|---:|
| Diversity ACGT vs AAAA | −22.85% | −24.54% | **+1.69%** |
| Repeat ATATATAT vs AAAATTTT | −34.08% | −34.34% | **+0.26%** |

**Interpretazione:**

- Entrambi i treatment-control producono effetti individuali enormi (~−22 a −36%) perché *qualsiasi* modifica del seed o PAM-proximal introduce mismatch nuovi
- Ma il **contrasto T-C è vicino a zero**: il modello dà predizioni quasi identiche a (1) ACGT e AAAA, (2) ATATATAT e AAAATTTT
- Per (1): conferma che il modello non rappresenta "diversità A/C/G/T" come concetto — vede 4 penalità posizionali indipendenti che si sommano in modo simile
- Per (2): conferma che il modello è strutturalmente cieco alle ripetizioni — due sequenze con stessa composizione globale ma struttura diversa danno output indistinguibili

Il "quadrante ideale 41.3%" delle guide per il repeat è un **falso segnale**: i `delta_TC` sono ≈ 0 (modello insensibile), quindi rientrano nei criteri (Δoff < 0 ⪅ 0, Δon ≥ −5% ⪅ 0) per pura assenza di risposta, non per efficacia.

**Implicazione metodologica:** queste due interventi servono da "stress test" diagnostico. Confermano operativamente che positional_mlp ha receptive field = 1 e non può rilevare proprietà joint o cross-position. Non è un fallimento del modello, è la conferma empirica del trade-off architetturale scelto in Run 15.

---

### F17 — Saturazione di pam_gate vs y_obs on-target

**Scoperta:** una volta corretta l'abduction in formula pam_gate-aware (F15, F-Fase 5), per la maggioranza delle guide GUIDEseq la variabile `U_on` risulta saturata al *clipping ceiling* di `logit(1 − ε) ≈ 16`:

```
U_on median (per-guida) = 14.18    U_on max = 15.43
```

**Causa diretta:** il modello ha appreso `pam_gate ≈ 0.73` (mean across pairs, std ≈ 0) per i PAM canonici NGG, mentre l'attività osservata on-target raggiunge il 99% (cap superiore di `reads_to_prob`). La formula di abduction:

```
U = logit(y_obs / pam_gate) − struct_logit
```

richiede `y_obs / pam_gate < 1`. Ma `99% / 73% = 1.35 > 1`, quindi il clipping interviene e U si satura. Il modello dice "il massimo di attività fisicamente possibile è 73%", i dati dicono 99% — inconsistenza strutturale.

**Conseguenza sui controfattuali on-target:** l'intervento `do(pos_14 = 0)` dovrebbe essere un *no-op* sull'on-target (la guida è uguale a se stessa, `P_14_factual ≈ 0` per match). Verifica nei dati: dove `U_on` non è saturato (es. AAVS1_site_13, U_on=1.40), `delta_on_p14 = 0` esattamente. Dove `U_on` è saturato (es. AAVS1_site_1, U_on=14.36), `delta_on_p14 = −26%` — un artefatto del clipping. La media per coppia (`delta_on_p14 = −7.99%`) è dominata dalle guide saturate; la mediana per-guida è 0.

**Perché non l'avevamo visto prima:** col vecchio codice che ignorava `pam_gate` nell'abduction (F15-bis: vecchia formula `U = logit(y_obs) − logit`), U assorbiva il gap modello-dati senza generare clipping. La saturazione era nascosta nell'approssimazione.

**Implicazione:** è un bug di calibrazione del modello, non dell'abduction. Va corretto trainando un modello con `pam_gate` calibrato sulla realtà dei dati on-target (vedi piano fase 7).

---

### F18 — I 4 limiti precisi di positional_mlp e cosa è già stato tentato per superarli

Sintesi delle limitazioni emerse dai findings precedenti:

| # | Limite | Origine | Già tentato di superarlo? | Esito |
|---|---|---|---|---|
| **L1** | Receptive field = 1 posizione (no joint/cross-position patterns) | `pos_node` applicata posizione-per-posizione con weight sharing | Sì — architetture regionali Exp04-11 | **Tutte hanno perso** contro positional_mlp |
| **L2** | Encoding 4-dim (perde l'identità delle basi: distingue solo match/wobble/transition/transversion) | Il branch positional_mlp ricalcola internamente un 4-dim invece di usare i 12-dim della `BiologicalMismatchEncoder` | No — l'encoder esiste ma il branch lo bypassa | Opportunità aperta |
| **L3** | Hard prior posizionale assume indipendenza | `w_pos[i]` indipendenti, nessun termine di interazione | No | Opportunità aperta |
| **L4** | PAM gate saturato (max 0.73 vs 99% osservato) | `pam_node` ha appreso valore conservativo durante training, vincolato dal training imbalanced | No — bug appena scoperto in F17 | Opportunità aperta, priorità immediata |

**Tabella architetture tentate per L1 (AUPRC test / GUIDEseq):**

| Run | Architettura | Cross-position | AUPRC test | AUPRC GUIDEseq |
|---|---|---|---:|---:|
| Exp04 | linear_bypass | No (somma mismatch) | 0.088 | 0.125 |
| Exp05 | mini_mlp | Regionale (binary mismatch) | 0.100 | 0.072 |
| Exp06 | typed_mlp | Regionale (typed mismatch) | 0.096 | 0.156 |
| Exp08 | learned_mlp | Regionale + embedding learnable | **0.008** | **0.007** |
| Exp09 | context_aware_mlp | Regionale + context-aware | **0.008** | 0.013 |
| Exp11 | typed_mlp + hybrid | Regionale | 0.090 | 0.166 |
| Exp12 | positional_mlp (v1) | No | 0.161 | 0.226 |
| Exp13 | positional_mlp + focal tune | No | 0.182 | 0.276 |
| **Exp15** | **positional_mlp + extended** | **No** | **0.154** | **0.285** |

**Lettura chiave:** tutte le architetture con capacità cross-position regionale (mini_mlp, typed_mlp, learned_mlp, context_aware_mlp) hanno *perso* contro il semplice positional_mlp. Quelle troppo capaci (learned, context_aware) sono crollate disastrosamente — overfitting puro sui ~3M esempi di training. La **parsimonia vince empiricamente**, non è solo una preferenza teorica.

**Conclusione operativa:** L1 e L3 sono "limiti by design" della filosofia parsimoniosa scelta in Run 15 — tentativi di superarli con maggiore capacità sono già stati fatti e hanno fallito. L2 e L4 sono invece opportunità non sfruttate che potrebbero essere risolte senza compromettere la filosofia del modello (entrambi mantengono 1-to-1 P_i ↔ posizione i).

---

### Todo Fase 7 — Piano sperimentale

- [x] **Run 18** — Fix calibrazione `pam_gate` (L4). Adottata l'alternativa (b): trasformazione `pam_gate` da fattore moltiplicativo a contributo additivo nel logit (vedi F19). Risultato: net win predittivo.
- [x] **Run 19** — Encoding 12-dim per `pos_node` (L2). Risultato: overfitting (vedi F20).
- [x] **Run 20** — Regolarizzazione causale soft (λ_causal=0.10) per arginare l'overfit di Run 19. Risultato: causal loss raw ridotta ma val AUPRC non migliorata (vedi F21).
- [ ] **(opzionale)** — Conv1D kernel=3 sulle posizioni (tentativo controllato per L1). Dopo i risultati F20-F21, l'evidenza empirica suggerisce che ogni aumento di capacità per posizione viene assorbito come overfit, quindi non è una priorità.
- [x] Aggiornato `simulate_intervention_batch.py` per supportare entrambe le modalità PAM (vedi F19, sezione "Aggiornamento abduction additiva").

---

### F19 — Run 18: PAM additivo risolve la saturazione di pam_gate

**Setup:** stesso modello di Run 15 (positional_mlp con encoding 4-dim ricalcolato internamente) ma con `pam_mode=additive`. La PAM contribuisce additivamente al logit invece di moltiplicare l'attività:

```
Multiplicative (Run 15):  activity = pam_gate × σ(struct_logit + U)        ← cap implicito a pam_gate
Additive (Run 18):        activity = σ(struct_logit + pam_logit + U)        ← nessun cap
```

Implementazione: nuovo parametro `pam_mode` in `NeuralSCM.__init__`, nuovo metodo `forward_logit()` in `PAMModule` che restituisce il logit raw, branching nel `_base_forward`. Il path multiplicativo è preservato per Run 15 backward-compat.

**Risultati predittivi (Run 15 vs Run 18):**

| Metrica | Run 15 (multiplicative) | Run 18 (additive) | Δ |
|---|---:|---:|---:|
| **CHANGE-seq test AUPRC** | **0.154** | **0.244** | **+59%** |
| CHANGE-seq test AUROC | 0.905 | 0.950 | +5% |
| **GUIDE-seq AUPRC** | **0.285** | **0.347** | **+22%** |
| GUIDE-seq AUROC | 0.964 | 0.977 | +1.3% |
| Loss finale (train) | 0.0029 | 0.0017 | −41% |
| L2 pesi finali | 14.5 | 9.2 | −37% |
| CCS_Overall | 0.333 | 0.167 | −50% |

**Interpretazione:**

Il `pam_gate` saturato a 0.73 (F17) era un **collo di bottiglia capacitivo**, non un regolarizzatore. Rimuovendo il cap implicito, il modello può raggiungere attività predette vicine a 1.0 per gli on-target, allineandosi con i dati. La saturazione di `U_on` nell'abduction pam_gate-aware sparisce (vedi sotto).

**Pesi posizionali Run 18** (più informativi/leggibili):

```
posizioni:  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16   17   18   19
peso:    -0.45 -0.57 -0.88 -1.10 -1.27 -1.03 -1.41 -1.40 -1.18 -0.97 -1.19 -1.36 -1.60 -1.88 -1.37 -1.68 -1.44 -1.67 -1.87 -1.52
```

Pattern biologicamente coerente: pos 0-3 PAM-distali poco penalizzate; seed (8-15) e PAM-proximal (16-19) fortemente penalizzate; pos 13 e 18 i picchi (coerente con la letteratura su seed criticality).

**Drop del CCS spiegato:** il CCS hardcoded usa `do(pam_gate=0.1)` per "PAM ablation". In modo additivo, `pam_logit_contrib=0.1` è un contributo logit quasi neutrale (non un'ablazione). R1 fallisce, R6 (gerarchia 1.0 > 0.2 > 0.1) regge. → 1/6 = 0.167. Non è una regressione del modello, è incompatibilità semantica del benchmark con la modalità additiva. Per renderlo significativo nel modo additivo servirebbe `do(pam_gate=-3.0)` (logit fortemente negativo).

**Aggiornamento abduction additiva:** lo script `simulate_intervention_batch.py` è stato aggiornato per supportare entrambe le modalità. In modalità additiva l'abduction è più semplice:
```
multiplicativo:  U = logit(y_obs / pam_gate) - struct_logit    (con clipping per y_obs > pam_gate)
additivo:        U = logit(y_obs) - final_logit                 (no division, no clipping artifact)
```

Con Run 18 la saturazione di U_on (F17) è **strutturalmente impossibile** — non c'è più un denominatore < 1 nella formula.

---

### F20 — Run 19: encoding 12-dim per pos_node causa overfitting

**Setup:** estende Run 18 (additive PAM) sostituendo l'input 4-dim di `pos_node` (mismatch type ricalcolato) con i 12-dim della `BiologicalMismatchEncoder`:

```
4-dim  (Run 18):  [mismatch_oh(4)]
12-dim (Run 19):  [mismatch_oh(4), sgRNA_base_oh(4), off_target_base_oh(4)]
```

Implementazione: nuovo parametro `positional_use_encoder: bool` in `NeuralSCM.__init__`, branching nel `_base_forward` del positional_mlp.

**Razionale teorico:** il 12-dim dovrebbe consentire preferenze posizionali base-specifiche (es. "C in pos 13 diverso da T in pos 13"), abilitando interventi semanticamente più ricchi (diversity ACGT, repeat) che il 4-dim non può rappresentare.

**Risultati predittivi (Run 18 vs Run 19):**

| Metrica | Run 18 (4-dim) | Run 19 (12-dim) | Δ |
|---|---:|---:|---:|
| Train AUPRC | 0.855 | 0.903 | +5.6% ↑ |
| Train AUROC | 0.984 | 0.990 | +0.6% ↑ |
| **Val AUPRC peak** | **0.115** | **0.077** | **−33% ↓** |
| **CHANGE-seq test AUPRC** | **0.244** | **0.171** | **−30% ↓** |
| GUIDE-seq AUPRC | 0.347 | 0.326 | −6% ↓ |
| Loss finale (train) | 0.0017 | 0.0013 | −24% |

**Diagnosi — overfitting netto:** Train sale, Val/Test scendono. Il modello con 12-dim ha più capacità per posizione (parametri della prima Linear: 40 → 104, ×2.6) e impara correlazioni base-specifiche presenti nel training set di CHANGE-seq che **non trasferiscono** a Val/Test/cross-assay.

**Pesi posizionali appiattiti:** Run 18 ha range −0.45 a −1.88 (varianza alta, signature seed forte). Run 19 ha range −0.22 a −1.03 (più piatti). Il segnale "questa posizione conta" è stato assorbito dentro la `pos_node` MLP arricchita, che lo apprende come pattern base-specifico training-specific. Effetto: il vincolo strutturale "peso per posizione" perde forza relativa.

**Implicazione metodologica per la tesi:** il bottleneck 4-dim **non è una limitazione**, è un **regolarizzatore strutturale**. Forza il modello a concentrarsi sull'unica feature biologicamente robusta (il tipo di mismatch) ignorando la composizione di base specifica (confounded con bias guide-specific nei dati). Il 4-dim batte il 12-dim non per parsimonia "filosofica" ma per inductive bias efficace.

---

### F21 — Run 20: regolarizzazione causale soft non sostituisce il bottleneck strutturale

**Setup:** estende Run 19 (additive PAM + 12-dim) alzando il peso della causal loss da `λ_causal=0.01` a `λ_causal=0.10` (×10). Razionale: rendere la directional margin loss un vincolo reale che possa controbilanciare l'overfit base-specifico.

**Risultati training (Run 19 vs Run 20, valori a epoch 11):**

| Componente | Run 19 (λ=0.01) | Run 20 (λ=0.10) | Δ |
|---|---:|---:|---:|
| `loss_pred` (Focal) | 0.0012 | 0.0014 | +17% |
| **`loss_causal` raw** | **0.0177** | **0.0121** | **−32%** |
| Contributo causal al totale (× λ) | 0.000177 | 0.00121 | ×6.8 |
| Loss totale | 0.0013 | 0.0026 | +100% |
| Train AUPRC | 0.904 | 0.884 | −2.2% |
| **Val AUPRC peak** | **0.077** | **~0.070** | **−9%** |

**Quello che funziona:** la regolarizzazione causale agisce sul suo target. `loss_causal_raw` scende del 32%, il modello fa **meno violazioni di direzionalità** durante il training. Train AUPRC scende leggermente, segno che il modello sta sacrificando un po' di fit per la causalità.

**Quello che NON funziona:** Val AUPRC peak è **leggermente più bassa** di Run 19 (0.070 vs 0.077). Il vincolo causale ha agito sulla coerenza direzionale dell'output globale ma **non sull'overfit specifico per posizione/base**.

**Diagnosi — dimensioni ortogonali:**

La causal loss attuale (`F.relu(delta_pred × −expected_direction + margin)`) misura *direzionalità globale* dell'attività post-intervento. L'overfit del 12-dim invece accade su preferenze posizionali base-specifiche. Il modello può soddisfare entrambi simultaneamente:

- Rispetta la monotonia: "più mismatch → meno attività" (causal loss soddisfatta)
- Apprende preferenze training-specific: "in pos 13 T penalizza di più di C" (overfit non vincolato dalla causal loss)

**Implicazione teorica:** la coerenza causale a livello di output non implica generalizzazione su dimensioni feature-specifiche. La **regolarizzazione strutturale (bottleneck 4-dim) agisce direttamente sulla capacità rappresentativa**, mentre la regolarizzazione causale soft agisce sul comportamento di output. Sono due tipi di vincolo che operano su livelli diversi del modello.

**Conclusione operativa:** Run 18 (additive PAM + 4-dim) resta il **modello vincitore**. L'evidenza empirica accumulata (Run 18 → 19 → 20) è coerente e ridondante: il bottleneck 4-dim è il regolarizzatore migliore disponibile in questo regime.

---

### Sintesi finale Fase 7 — modello vincitore e configurazione consolidata

| Componente | Scelta finale | Run di riferimento |
|---|---|---|
| Architettura | positional_mlp (20 nodi P_i indipendenti) | Run 15-20 |
| Encoder per pos_node | 4-dim mismatch type ricalcolato internamente | Run 15, 18 |
| PAM gating | additivo (contributo logit) | **Run 18** |
| Encoding ricco (12-dim) | NON adottato (overfit) | Run 19 (negativo) |
| Regolarizzazione causale forte | NON necessaria | Run 20 (negativo) |
| λ_causal | 0.01 (decorativo nel positional_mlp) | Run 15-18 |
| Abduction | algebraica post-hoc, formula additiva | F15 → F19 |
| **Modello di riferimento per explainability** | **Exp18_Positional_AdditivePAM** | F19 |

L'explainability finale (intervento truncation, do(pos_14), diversity, repeat) verrà eseguita sul modello Exp18 con `simulate_intervention_batch.py` aggiornato per la modalità additiva.
