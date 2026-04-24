Fase 1 (già fatta): feature extractor DAG + XGBoost
    → baseline associazionale, gradino 1

Fase 2 (prossima): SCM classico
    → stima parametri equazioni strutturali
    → risponde a domande do() con adjustment formula
    → valida implicazioni causali del DAG sui dati
    → dimostra che il gradino 1 è insufficiente

Fase 3: Neural SCM
    → implementa ogni equazione come modulo neurale
    → training standard su CHANGE-seq
    → valutazione predittiva (AUPRC/AUROC) → deve competere con CCLMoff
    → valutazione causale (CCS, controfattuali biologicamente plausibili)
    → questo è il contributo principale della tesi