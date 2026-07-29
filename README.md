# Semantic Multiplexing Gain in Wireless Systems via Expanded Embeddings: A BERT Case Study

Code and stored results for the IEEE Communications Letters submission
by Ki-Ho Lee, Hyun-Ho Choi, and Jung-Ryun Lee.

Multiple users share one expanded embedding block of dimension
`d_s = K * d_b`: each user's frozen BERT sentence embedding is projected
into the shared space, superimposed through learnable masks, and
demultiplexed by user-wise attention. All reported transceivers are
trained with SNR-aware MAML; a joint multi-SNR training reference is
included.

## Files

| File | Purpose |
|---|---|
| `bert_semcom.py` | Shared library: BERT extractor, transceiver model, channel, MAML helpers |
| `cl_experiments.py` | Held-out split, joint-trained configurations, ToDMA-style token-domain benchmark, linear probe, latency |
| `cl_maml_all.py` | SNR-aware MAML training for every reported configuration (including the conventional orthogonal scheme) |
| `cl_maml_extra.py` | MAML K sweep (K = 1, 2, 8) and DistilBERT replication |
| `replot_cl.py` | Regenerates both result figures from the stored JSON results |
| `fig_cl/*.json`, `fig_cl/*.csv` | Stored raw results behind every figure and quoted number |

## Reproducing

Requirements: Python 3.10+, PyTorch (CUDA), `transformers`, `datasets`,
`matplotlib`, `numpy`. AG News loads from the Hugging Face hub
(`fancyzhx/ag_news` fallback included).

```bash
python cl_experiments.py --save-dir fig_cl   # joint runs + ToDMA benchmark (~3 h on a laptop GPU)
python cl_maml_all.py    --save-dir fig_cl   # MAML runs (~9 h)
python cl_maml_extra.py  --save-dir fig_cl   # MAML K sweep + DistilBERT (~6 h)
python replot_cl.py                          # figures from stored results
```

All experiments fix their random seeds (training seed 42, evaluation
seed 123, ToDMA seed 7) and evaluate on a held-out test split of 2,000
AG News sentences disjoint from the 8,000-sentence training pool.
`replot_cl.py` reads only the stored results, so the figures are
regenerable without rerunning the experiments.
