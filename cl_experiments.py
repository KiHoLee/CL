# =========================================================
# cl_experiments.py — IEEE Communications Letters revision
#
# New experiments addressing TVT reviewer comments:
#   R2-2 : held-out train/test split (8000/2000, disjoint)
#   R1-2 : bandwidth-expansion sweep K = 1, 2, 4, 8 at U = 4
#   R1-5c: static random-projection mask baseline (frozen masks)
#   R2-3 : ToDMA-style token-domain CS baseline (same channel budget)
#   R1-5b: DistilBERT generalization check
#   R2-4 : downstream AG News topic accuracy (linear probe)
#   R1-3 : runtime latency / parameter count
#
# All evaluations are on the held-out test split.
# Outputs: fig_cl/cl_results.json, fig_cl/cl_convergence.csv
# =========================================================

import argparse, os, json, csv, math, random, time, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_semcom import (
    BertSemComMux, semantic_loss,
    split_params, select_inner_keys, gather_inner_params,
    apply_inner_update
)

EVAL_SNRS = (0, 5, 10, 15, 20, 25, 30)
TRAIN_SNRS = [0, 5, 10, 15, 20, 25]


def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------
# Data with labels + disjoint split
# ---------------------------------------------------------
def load_agnews_labeled(n_train=8000, n_test=2000,
                        min_len=5, max_len=30, seed=42):
    from datasets import load_dataset
    try:
        ds = load_dataset("ag_news", split="train")
    except Exception:
        ds = load_dataset("fancyzhx/ag_news", split="train")
    items = []
    for ex in ds:
        first = ex["text"].split(".")[0].strip()
        w = first.split()
        if min_len <= len(w) <= max_len:
            items.append((first, int(ex["label"])))
        if len(items) >= (n_train + n_test):
            break
    rng = random.Random(seed)
    rng.shuffle(items)
    train = items[:n_train]
    test = items[n_train:n_train + n_test]
    return train, test


class Extractor:
    """Frozen encoder (BERT or DistilBERT), mean-pooled."""
    def __init__(self, model_name, device):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.embed_dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts, max_length=64):
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=max_length,
                                return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        h = out.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)


class SplitCache:
    """Train/test embedding caches. Centering mean computed on the
    TRAIN pool only and reused for the test split (R2-2)."""
    def __init__(self, extractor, train_items, test_items, bs=64):
        self.train_texts = [t for t, _ in train_items]
        self.train_labels = torch.tensor([l for _, l in train_items])
        self.test_texts = [t for t, _ in test_items]
        self.test_labels = torch.tensor([l for _, l in test_items])

        def enc_all(texts):
            embs = []
            for i in range(0, len(texts), bs):
                embs.append(extractor.encode(texts[i:i + bs]).cpu())
            return torch.cat(embs, 0)

        print(f"[INFO] Encoding {len(self.train_texts)} train sentences...",
              flush=True)
        E_tr = enc_all(self.train_texts)
        print(f"[INFO] Encoding {len(self.test_texts)} test sentences...",
              flush=True)
        E_te = enc_all(self.test_texts)

        self.mu = E_tr.mean(0, keepdim=True)
        self.train = F.normalize(E_tr - self.mu, p=2, dim=-1)
        self.test = F.normalize(E_te - self.mu, p=2, dim=-1)

        cs = F.cosine_similarity(
            self.test[torch.randperm(len(self.test))[:500]],
            self.test[torch.randperm(len(self.test))[:500]], dim=-1)
        print(f"[INFO] test split: random-pair cos mean={cs.mean():.4f} "
              f"std={cs.std():.4f}", flush=True)

    def sample_train(self, U, device):
        idx = torch.randint(0, len(self.train), (U,))
        return self.train[idx].to(device), idx

    def sample_test(self, U, device, gen=None):
        idx = torch.randint(0, len(self.test), (U,), generator=gen)
        return self.test[idx].to(device), idx


# ---------------------------------------------------------
# Training (train pool) + held-out evaluation (test pool)
# ---------------------------------------------------------
def train_config(cache, U, d_bert, K, device, mode="joint",
                 freeze_masks=False, epochs=200, steps=300,
                 lr=1e-3, inner_lr=5e-4, inner_steps=1, meta_batch=4,
                 lam=0.5, channel="rayleigh", conv_trials=50,
                 label="", seed=42):
    set_seed(seed)
    d_shared = d_bert * K
    model = BertSemComMux(U, d_bert, d_shared, 512).to(device)
    if freeze_masks:
        model.user_mask.weight.requires_grad_(False)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    inner_keys = select_inner_keys(mode)
    conv = []
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for _ in range(steps):
            if mode == "joint":
                b, _ = cache.sample_train(U, device)
                snr = float(random.choice(TRAIN_SNRS))
                b_hat = model(b, snr, channel)
                loss, _, _ = semantic_loss(b, b_hat, lam)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_sum += loss.item()
            else:
                snr_tasks = [float(x) for x in
                             random.sample(TRAIN_SNRS, meta_batch)]
                base_params = split_params(model)
                meta_loss = torch.tensor(0.0, device=device)
                for snr in snr_tasks:
                    fp = {k: {n: p for n, p in v.items()}
                          for k, v in base_params.items()}
                    for _ in range(inner_steps):
                        b_s, _ = cache.sample_train(U, device)
                        bh = model(b_s, snr, channel, params=fp)
                        ls, _, _ = semantic_loss(b_s, bh, lam)
                        fl, mi = gather_inner_params(fp, inner_keys)
                        grads = torch.autograd.grad(ls, fl)
                        fp = apply_inner_update(fp, inner_keys, mi, grads,
                                                inner_lr, "first")
                    b_q, _ = cache.sample_train(U, device)
                    bh_q = model(b_q, snr, channel, params=fp)
                    ql, _, _ = semantic_loss(b_q, bh_q, lam)
                    meta_loss = meta_loss + ql
                meta_loss = meta_loss / float(meta_batch)
                opt.zero_grad(set_to_none=True)
                meta_loss.backward()
                opt.step()
                loss_sum += meta_loss.item()

        # light held-out convergence eval @10 dB
        model.eval()
        with torch.no_grad():
            c = 0.0
            for _ in range(conv_trials):
                b, _ = cache.sample_test(U, device)
                bh = model(b, 10, channel)
                c += F.cosine_similarity(bh, b, dim=-1).mean().item()
        conv.append({"epoch": ep, "loss": loss_sum / steps,
                     "cos10": c / conv_trials})
        if ep % 20 == 0 or ep == 1 or ep == epochs:
            print(f"  [{label} Ep {ep:03d}/{epochs}] "
                  f"loss={loss_sum/steps:.4f} "
                  f"cos@10dB(test)={c/conv_trials:.4f}", flush=True)

    train_time = time.time() - t0
    return model, conv, train_time


@torch.no_grad()
def final_eval(model, cache, U, device, channel="rayleigh",
               trials=500, collect_at=None, seed=123):
    """Held-out SNR sweep. If collect_at is set (snr list), also return
    recovered embeddings + label indices for the linear probe."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    out = {}
    collected = {}
    for snr in EVAL_SNRS:
        ct, mt = 0.0, 0.0
        rec, idxs = [], []
        for _ in range(trials):
            b, idx = cache.sample_test(U, device, gen=gen)
            bh = model(b, snr, channel)
            ct += F.cosine_similarity(bh, b, dim=-1).mean().item()
            mt += F.mse_loss(bh, b).item()
            if collect_at and snr in collect_at:
                rec.append(bh.cpu())
                idxs.append(idx)
        out[snr] = {"cos": ct / trials, "mse": mt / trials}
        if collect_at and snr in collect_at:
            collected[snr] = (torch.cat(rec, 0), torch.cat(idxs, 0))
    return out, collected


# ---------------------------------------------------------
# Linear probe (AG News 4-class) on clean train embeddings
# ---------------------------------------------------------
def train_probe(cache, device, epochs=300, lr=1e-2):
    X = cache.train.to(device)
    y = cache.train_labels.to(device)
    W = nn.Linear(X.shape[1], 4).to(device)
    opt = torch.optim.Adam(W.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(W(X), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc_clean = (W(cache.test.to(device)).argmax(-1).cpu()
                     == cache.test_labels).float().mean().item()
    print(f"[INFO] probe clean test accuracy = {acc_clean:.4f}", flush=True)
    return W, acc_clean


@torch.no_grad()
def probe_accuracy(W, collected, cache, device):
    accs = {}
    for snr, (rec, idx) in collected.items():
        pred = W(rec.to(device)).argmax(-1).cpu()
        accs[snr] = (pred == cache.test_labels[idx]).float().mean().item()
    return accs


# ---------------------------------------------------------
# ToDMA-style token-domain CS baseline (R2-3)
#   - shared random Gaussian codebook over the BERT vocabulary
#   - per-slot OMP detection (U iterations)
#   - genie-aided token-source association (upper bound):
#     a user's token is recovered iff it lies in the detected support
#   - identical total channel budget: T * L = d_s = 3072 real uses
#   - identical per-dimension sum power = 1 and noise var = 1/gamma
# ---------------------------------------------------------
@torch.no_grad()
def todma_eval(extractor, cache, device, U=4, T=24, L=128,
               n_frames=200, channel="rayleigh", seed=7,
               collect_at=None):
    tok = extractor.tokenizer
    V = tok.vocab_size
    g = torch.Generator().manual_seed(seed)
    C = torch.randn(V, L, generator=g)
    C = F.normalize(C, p=2, dim=1).to(device)          # unit-norm atoms
    amp = math.sqrt(L / U)                              # per-user energy

    # pre-tokenize test sentences (no special tokens), truncate to T
    tok_ids = [tok(t, add_special_tokens=False)["input_ids"][:T]
               for t in cache.test_texts]

    results = {}
    collected = {}
    t_omp_total, n_omp = 0.0, 0
    for snr in EVAL_SNRS:
        sigma = math.sqrt(1.0 / (10 ** (snr / 10.0)))
        cs_sum, n_sent = 0.0, 0
        tok_err_sum, tok_cnt = 0, 0
        rec_all, idx_all = [], []
        rng = torch.Generator().manual_seed(seed + snr)
        for fr in range(n_frames):
            idx = torch.randint(0, len(cache.test), (U,), generator=rng)
            seqs = [tok_ids[i] for i in idx.tolist()]
            if channel == "rayleigh":
                hr = torch.randn(U, generator=rng)
                hi = torch.randn(U, generator=rng)
                h = torch.sqrt(hr ** 2 + hi ** 2) / math.sqrt(2.0)
            else:
                h = torch.ones(U)
            h = h.to(device)

            det_ids = [[] for _ in range(U)]
            t1 = time.time()
            for t in range(T):
                active = [(u, seqs[u][t]) for u in range(U)
                          if t < len(seqs[u])]
                if not active:
                    break
                y = torch.zeros(L, device=device)
                for u, tid in active:
                    y = y + h[u] * amp * C[tid]
                y = y + sigma * torch.randn(L, device=device)
                # OMP: U iterations
                residual = y.clone()
                support = []
                for _ in range(min(U, len(active))):
                    corr = torch.mv(C, residual).abs()
                    if support:
                        corr[torch.tensor(support, device=device)] = -1
                    k = int(corr.argmax().item())
                    support.append(k)
                    A = C[support].T * amp                  # (L, |S|)
                    coef, *_ = torch.linalg.lstsq(A, y.unsqueeze(1))
                    residual = y - (A @ coef).squeeze(1)
                sset = set(support)
                for u, tid in active:
                    tok_cnt += 1
                    if tid in sset:
                        det_ids[u].append(tid)              # genie assoc.
                    else:
                        tok_err_sum += 1                    # erasure
            t_omp_total += time.time() - t1
            n_omp += 1

            texts = [tok.decode(d) if d else "[UNK]" for d in det_ids]
            emb = extractor.encode(texts).cpu()
            emb = F.normalize(emb - cache.mu, p=2, dim=-1)
            ref = cache.test[idx]
            cs_sum += F.cosine_similarity(emb, ref, dim=-1).sum().item()
            n_sent += U
            if collect_at and snr in collect_at:
                rec_all.append(emb)
                idx_all.append(idx)

        results[snr] = {"cos": cs_sum / n_sent,
                        "token_err": tok_err_sum / max(tok_cnt, 1)}
        if collect_at and snr in collect_at:
            collected[snr] = (torch.cat(rec_all, 0), torch.cat(idx_all, 0))
        print(f"  [ToDMA T={T} L={L}] SNR={snr} "
              f"cos={results[snr]['cos']:.4f} "
              f"tokErr={results[snr]['token_err']:.4f}", flush=True)

    results["omp_ms_per_frame"] = 1000.0 * t_omp_total / max(n_omp, 1)
    return results, collected


# ---------------------------------------------------------
# Latency / parameter count (R1-3)
# ---------------------------------------------------------
@torch.no_grad()
def measure_latency(model, cache, U, device, n=200):
    b, _ = cache.sample_test(U, device)
    for _ in range(20):
        model(b, 10, "rayleigh")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        model(b, 10, "rayleigh")
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = 1000.0 * (time.time() - t0) / n
    n_params = sum(p.numel() for p in model.parameters())
    return ms, n_params


@torch.no_grad()
def measure_bert_latency(extractor, texts, n=50):
    for _ in range(5):
        extractor.encode(texts[:4])
    if extractor.device == "cuda" or (hasattr(extractor.device, "type")
                                      and extractor.device.type == "cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    for i in range(n):
        extractor.encode([texts[i % len(texts)]])
    torch.cuda.synchronize()
    return 1000.0 * (time.time() - t0) / n


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", default="fig_cl")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--todma-frames", type=int, default=200)
    ap.add_argument("--skip-distil", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[Device: {device}]", flush=True)

    train_items, test_items = load_agnews_labeled()
    print(f"[INFO] split: {len(train_items)} train / "
          f"{len(test_items)} test", flush=True)

    bert = Extractor("bert-base-uncased", device)
    d_bert = bert.embed_dim
    cache = SplitCache(bert, train_items, test_items)

    R = {"meta": {"epochs": args.epochs, "steps": args.steps,
                  "trials": args.trials}}
    conv_rows = []

    # linear probe on clean train embeddings
    probe, acc_clean = train_probe(cache, device)
    R["probe_clean_acc"] = acc_clean

    PROBE_SNRS = list(EVAL_SNRS)

    # ---- (1) orthogonal baseline U=1, d=768 ----
    # ---- (2) proposed U=1..6, K=4 ----
    # ---- (3) K sweep U=4, K in {1,2,8} ----
    # ---- (4) random frozen masks U=4, K=4 ----
    # ---- (5) MAML full U=4, K=4 ----
    configs = [
        ("baseline_U1_K1", dict(U=1, K=1)),
        ("prop_U1_K4", dict(U=1, K=4)),
        ("prop_U2_K4", dict(U=2, K=4)),
        ("prop_U3_K4", dict(U=3, K=4)),
        ("prop_U4_K4", dict(U=4, K=4)),
        ("prop_U5_K4", dict(U=5, K=4)),
        ("prop_U6_K4", dict(U=6, K=4)),
        ("ksweep_U4_K1", dict(U=4, K=1)),
        ("ksweep_U4_K2", dict(U=4, K=2)),
        ("ksweep_U4_K8", dict(U=4, K=8)),
        ("randmask_U4_K4", dict(U=4, K=4, freeze_masks=True)),
        ("maml_U4_K4", dict(U=4, K=4, mode="maml_full")),
    ]

    masks_store = {}
    for name, kw in configs:
        print(f"\n=== {name} ===", flush=True)
        U, K = kw.pop("U"), kw.pop("K")
        model, conv, ttime = train_config(
            cache, U, d_bert, K, device,
            epochs=args.epochs, steps=args.steps, label=name, **kw)
        collect = PROBE_SNRS if name in (
            "baseline_U1_K1", "prop_U4_K4", "randmask_U4_K4") else None
        sweep, collected = final_eval(model, cache, U, device,
                                      trials=args.trials,
                                      collect_at=collect)
        entry = {"U": U, "K": K, "train_s": ttime,
                 "snr": {str(s): sweep[s] for s in EVAL_SNRS}}
        if collect:
            entry["probe_acc"] = {str(s): a for s, a in
                                  probe_accuracy(probe, collected,
                                                 cache, device).items()}
        if name in ("prop_U2_K4", "prop_U3_K4", "prop_U4_K4"):
            m = model.user_mask.weight.detach().cpu()
            mn = F.normalize(m, p=2, dim=1)
            masks_store[name] = (mn @ mn.T).numpy().tolist()
        if name == "prop_U4_K4":
            ms_gpu, n_params = measure_latency(model, cache, U, device)
            cpu_model = BertSemComMux(U, d_bert, d_bert * K, 512)
            cpu_model.load_state_dict(model.state_dict())
            cpu_dev = torch.device("cpu")
            b_cpu, _ = cache.sample_test(U, cpu_dev)
            for _ in range(10):
                cpu_model(b_cpu, 10, "rayleigh")
            t0 = time.time()
            for _ in range(50):
                cpu_model(b_cpu, 10, "rayleigh")
            ms_cpu = 1000.0 * (time.time() - t0) / 50
            R["latency"] = {"proposed_gpu_ms": ms_gpu,
                            "proposed_cpu_ms": ms_cpu,
                            "params": n_params}
            print(f"[LATENCY] proposed frame: {ms_gpu:.2f} ms (GPU) "
                  f"{ms_cpu:.2f} ms (CPU), params={n_params/1e6:.2f}M",
                  flush=True)
        R[name] = entry
        for c in conv:
            conv_rows.append([name, c["epoch"], c["loss"], c["cos10"]])
        with open(os.path.join(args.save_dir, "cl_results.json"), "w") as f:
            json.dump(R, f, indent=1)

    R["mask_corr"] = masks_store

    # ---- (6) ToDMA-style baseline, two budget splits ----
    print("\n=== ToDMA-style baseline ===", flush=True)
    R["bert_tx_ms"] = measure_bert_latency(bert, cache.test_texts)
    print(f"[LATENCY] BERT encode per sentence: {R['bert_tx_ms']:.1f} ms",
          flush=True)
    for (T, L) in [(24, 128), (16, 192)]:
        res, coll = todma_eval(bert, cache, device, U=4, T=T, L=L,
                               n_frames=args.todma_frames,
                               collect_at=PROBE_SNRS if T == 24 else None)
        key = f"todma_T{T}_L{L}"
        R[key] = {str(s): res[s] for s in EVAL_SNRS}
        R[key]["omp_ms_per_frame"] = res["omp_ms_per_frame"]
        if coll:
            R[key]["probe_acc"] = {str(s): a for s, a in
                                   probe_accuracy(probe, coll,
                                                  cache, device).items()}
    with open(os.path.join(args.save_dir, "cl_results.json"), "w") as f:
        json.dump(R, f, indent=1)

    # ---- (7) DistilBERT generalization check ----
    if not args.skip_distil:
        print("\n=== DistilBERT check (U=4, K=4) ===", flush=True)
        distil = Extractor("distilbert-base-uncased", device)
        dcache = SplitCache(distil, train_items, test_items)
        model, conv, _ = train_config(dcache, 4, distil.embed_dim, 4,
                                      device, epochs=args.epochs,
                                      steps=args.steps, label="distil")
        sweep, _ = final_eval(model, dcache, 4, device, trials=args.trials)
        R["distil_U4_K4"] = {"snr": {str(s): sweep[s] for s in EVAL_SNRS}}
        for c in conv:
            conv_rows.append(["distil_U4_K4", c["epoch"], c["loss"],
                              c["cos10"]])

    with open(os.path.join(args.save_dir, "cl_results.json"), "w") as f:
        json.dump(R, f, indent=1)
    with open(os.path.join(args.save_dir, "cl_convergence.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "epoch", "loss", "cos10_test"])
        w.writerows(conv_rows)
    print("\nAll CL experiments complete.", flush=True)


if __name__ == "__main__":
    main()
