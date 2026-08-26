# Semantic Multiplexing Gain in Wireless Systems via Expanded Embeddings: A BERT Case Study

Code, stored results, and supplementary material for the IEEE
Communications Letters submission by Ki-Ho Lee, Hyun-Ho Choi, and
Jung-Ryun Lee.

Multiple users share one expanded embedding block of dimension
`d_s = K * d_b`. Each user's frozen BERT sentence embedding is projected
into the shared space, superimposed through learnable masks, and
demultiplexed by user-wise attention. Every transceiver reported in the
letter, including the conventional orthogonal scheme, is trained with
SNR-aware MAML; the same architecture trained without MAML, as in the
earlier JSAC paper, is included as a prior-art reference and is labelled
"Training w/o MAML [5]" in the figures.

## Files

| File | Purpose |
|---|---|
| `bert_semcom.py` | Shared library: BERT extractor, transceiver model, channel, MAML helpers |
| `cl_experiments.py` | Held-out split, configurations trained without MAML, token-domain (ToDMA) comparison, linear probe, latency |
| `cl_maml_all.py` | SNR-aware MAML training for every reported configuration, including the conventional orthogonal scheme |
| `cl_maml_extra.py` | MAML expansion-factor sweep (K = 1, 2, 8) and DistilBERT replication |
| `cl_todma_load.py` | Evaluates the token-domain scheme at U = 1, 2, 3, 5, 6 for the load sweep of Fig. 3 |
| `replot_cl.py` | Regenerates Figs. 2 and 3 of the letter from the stored JSON results |
| `probe_vs_cosine.py` | Supplementary probe-accuracy-versus-cosine-similarity analysis |
| `fig_cl/*.json`, `fig_cl/*.csv` | Stored raw results behind every figure and every number quoted in the letter |

## Reproducing

Requirements: Python 3.10+, PyTorch (CUDA), `transformers`, `datasets`,
`matplotlib`, `numpy`. AG News loads from the Hugging Face hub
(`fancyzhx/ag_news` fallback included).

```bash
python cl_experiments.py --save-dir fig_cl   # runs without MAML + token-domain comparison (~3 h on a laptop GPU)
python cl_maml_all.py    --save-dir fig_cl   # MAML runs (~9 h)
python cl_maml_extra.py  --save-dir fig_cl   # MAML K sweep + DistilBERT (~6 h)
python cl_todma_load.py  --save-dir fig_cl   # token-domain load sweep (evaluation only)
python replot_cl.py                          # Figs. 2 and 3 from the stored results
python probe_vs_cosine.py                    # supplementary analysis below
```

All experiments fix their random seeds (training seed 42, evaluation
seed 123, token-domain seed 7) and evaluate on a held-out test split of
2,000 AG News sentences disjoint from the 8,000-sentence training pool.
The centering mean, the transceiver parameters, and the linear probe are
fitted on the training pool only. `replot_cl.py` and
`probe_vs_cosine.py` read only the stored results, so every figure is
regenerable without rerunning the experiments.

## Figures of the letter

**Fig. 2 - per-user cosine similarity vs. SNR.** The proposed scheme for
U = 1..4 at K = 4, the conventional orthogonal scheme, the two
matched-budget comparison schemes (random-projection mask and the
token-domain scheme at two slot/codeword splits), and the same
architecture trained without MAML:

![Fig. 2](fig_cl/cl_fig_mux.png)

**Fig. 3 - aggregate fidelity across load at 10 dB.** Three bars per
load point (proposed, training without MAML, and the token-domain
scheme at 24x128), the conventional single-user block on the left, and
the fully loaded orthogonal aggregate as the dash-dotted reference:

![Fig. 3](fig_cl/cl_fig_agg.png)

At full load the shared block stays within 1% of the fully loaded
orthogonal aggregate over the same 3072 channel uses (3.59 against
3.62), and under overload it reaches 1.45 times that aggregate at
U = 6.

## Supplementary: probe accuracy vs. cosine similarity

The letter measures semantic fidelity by the cosine similarity of the
recovered embeddings and corroborates it with a downstream perception
metric, namely the AG News topic accuracy of a linear probe fitted on
the training pool and applied to the recovered test embeddings. The
noiseless reference is about 0.854 and the sampling error is about
+/-0.01.

Across 7 schemes x 7 SNRs (49 operating points), probe accuracy tracks
cosine similarity with a Pearson correlation of **r = 0.903**:

![Probe accuracy vs. cosine similarity](fig_cl/probe_vs_cosine.png)

Two readings follow. First, the low-SNR advantage of the embedding
schemes over the token-domain scheme appears in both metrics, for
example 0.848 against 0.772 in cosine similarity and 0.805 against
0.762 in accuracy at 5 dB. Second, schemes within about 0.01 of each
other in cosine similarity differ in accuracy by at most 0.023, which
is within about two standard errors of the sampling noise, so the
cosine metric used throughout the letter is consistent with downstream
perception on this task.

## Citation

The letter is under review. Until it appears, please cite this
repository together with the earlier shared-embedding paper:

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Transformer-based shared
> embedding for multiple access in semantic communications," *IEEE J.
> Sel. Areas Commun.*, vol. 44, pp. 2622-2637, 2026.
