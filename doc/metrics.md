# Metrics

## Causal Consistency Score (CCS)

Il **Causal Consistency Score (CCS)** misura quanto le predizioni del modello rispettano una serie di regole causali/biologiche attese sotto interventi sintetici controllati su sequenza e PAM.

In pratica:
- si costruisce una baseline per ogni guida (match perfetto con PAM canonico `AGG`),
- si applicano mutazioni/interventi specifici,
- si verifica se la probabilita predetta cambia nella direzione attesa,
- il CCS è la frazione di esempi che soddisfa **tutte** le regole richieste.

## Modalita supportate

- `3_rules`: usa 3 regole base.
- `6_rules`: usa 6 regole (le 3 base + 3 aggiuntive).

Nel codice attuale (`evaluation/ccs.py`):
- in `3_rules`: `CCS_Overall = mean(R1 & R2 & R3)`
- in `6_rules`: `CCS_Overall = mean(R1 & R2 & R3 & R4 & R5 & R6)`

## Output della metrica

`calculate_ccs(...)` restituisce un dizionario con:
- score per singola regola (`R1_*`, `R2_*`, ...)
- `CCS_Overall`

Ogni score e una media di variabili binarie di pass/fail.

## Range e valori possibili

### Range teorico

Tutti gli score (incl. `CCS_Overall`) sono in:

$$
[0,1]
$$

### Valori possibili su dataset finito

Con $N$ guide, ogni score e una media di 0/1, quindi assume valori discreti:

$$
\left\{0, \frac{1}{N}, \frac{2}{N}, \dots, 1\right\}
$$

## Interpretazione pratica

- `CCS_Overall = 1.0`: tutte le guide rispettano tutte le regole richieste.
- `CCS_Overall = 0.0`: nessuna guida rispetta simultaneamente tutte le regole.
- valori intermedi: percentuale di guide coerenti con il set completo di vincoli.

Lettura operativa (euristica):
- `[0.00, 0.30)`: coerenza causale molto bassa
- `[0.30, 0.60)`: coerenza parziale
- `[0.60, 0.85)`: buona coerenza
- `[0.85, 1.00]`: coerenza alta/molto alta

Nota: la soglia di accettabilita va tarata sul contesto sperimentale e confrontata con baseline storiche del progetto.

## Differenza tra score per regola e CCS complessivo

- Le metriche `Rk_*` mostrano il comportamento su singolo vincolo.
- `CCS_Overall` e piu severo: richiede il rispetto simultaneo di tutte le regole della modalita scelta.

Conseguenza: `CCS_Overall` e normalmente minore o uguale a ciascun `Rk_*`.

## Requisiti input e robustezza

L'implementazione corrente valida esplicitamente che:
- `mode` sia `3_rules` o `6_rules`
- `unique_guides` non sia vuoto
- ogni guida abbia almeno 20 nt
- `predict_fn` ritorni output 1D o 2D interpretabile come probabilita della classe positiva
- il numero di predizioni sia coerente con il numero di input
- non ci siano valori `NaN` o infiniti

## Caveat di interpretazione

- CCS misura **consistenza direzionale** rispetto a regole predefinite, non calibrazione probabilistica.
- Un modello puo avere AUPRC/AUROC alti ma CCS basso (buona associazione, bassa coerenza causale).
- Un CCS alto non sostituisce test interventionali reali: e una verifica in-silico di plausibilita causale.
