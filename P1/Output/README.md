# P1/Output Structure

Phase-grouped layout for OpTC H051 thesis artifacts.

```
P1/Output/
├── results_phase0/      # Detector layer: Velox, ORTHRUS, Flash, Magic
│                        # Recall@K, precision ceiling, threshold sweep
├── results_phase1/      # Velox optimization (chunked loading T1-T3)
│                        # + Tier-2 pseudocode for paper
├── results_phase2/      # Tier-2 SLM experiments (RQ1a, RQ1b)
│                        # LoRA 0.5B, BERT 150M, Distill TinyBERT 4M, TTP unseen
│                        # + Reports for advisor + go/no-go decision
├── data/                # Input data + labels
│                        # alerts-enriched-v2.jsonl, gt_and_scores.json,
│                        # ttp_holdout.jsonl (synthetic TTP chains from ART YAML)
├── models/              # Trained model weights + tokenizer + config
│                        # (lora-05b/, bert-150m/, tinybert-4m-distilled/)
├── logs/                # Training/inference logs (gitignored)
├── benchmarks/          # HW latency sweeps (CSV/JSON)
└── (root)               # Empty - kept for cleanliness
```

## Phase summary

- **Phase 0** (detectors): Compare 4 PIDS on OpTC H051 full test
  - Velox AUC 0.917 > ORTHRUS AUC 0.888 > Magic AUC 0.751 > Flash AUC 0.564
  - Recall@10K ceiling ~30% (80/114 GT nodes inactive in test)

- **Phase 1** (Velox optimization): Chunked lazy-load, GPU bf16, TP-aware split
  - Velox T1-T3: epoch -39%, RAM 28GB → 3.5GB, no swap

- **Phase 2** (Tier-2 SLM): Note.md RQ1a/RQ1b
  - H0 (encoder >= SLM) confirmed on V2 and TTP unseen
  - Distill down to 4M fails (shortcut leak from imbalanced teacher)
  - Final verdict: encoder 150M (best, AP 0.9999 V2 / 0.6603 TTP) > TF-IDF (AP 0.89) >> SLM

See `results_phase2/go-nogo-decision.md` for detailed analysis.