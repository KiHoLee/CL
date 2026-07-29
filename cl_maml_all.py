# =========================================================
# cl_maml_all.py — promote SNR-aware MAML to the default
# training procedure for all reported configurations.
#
# Adds MAML-trained counterparts of the load sweep, the
# conventional orthogonal scheme (fairness: both sides at
# their best), and the random-mask variant. The K sweep and
# DistilBERT stay joint-trained (sensitivity studies).
#
# Held-out evaluation identical to cl_experiments.py.
# Output: fig_cl/cl_results_maml.json (+ convergence CSV)
# =========================================================

import argparse, os, json, csv, time, random
import torch
import torch.nn.functional as F

from bert_semcom import (
    BertSemComMux, semantic_loss, split_params,
    gather_inner_params, apply_inner_update
)
from cl_experiments import (
    load_agnews_labeled, Extractor, SplitCache,
    final_eval, train_probe, probe_accuracy, set_seed,
    EVAL_SNRS, TRAIN_SNRS, measure_latency
)


def train_maml(cache, U, d_bert, K, device, freeze_masks=False,
               epochs=200, steps=300, lr=1e-3, inner_lr=5e-4,
               meta_batch=4, lam=0.5, channel="rayleigh",
               conv_trials=50, label="", seed=42):
    set_seed(seed)
    model = BertSemComMux(U, d_bert, d_bert * K, 512).to(device)
    if freeze_masks:
        model.user_mask.weight.requires_grad_(False)
        inner_keys = ["tx_proj", "user_query", "rx_proj"]
    else:
        inner_keys = ["tx_proj", "user_mask", "user_query", "rx_proj"]
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    conv = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for _ in range(steps):
            snr_tasks = [float(x) for x in
                         random.sample(TRAIN_SNRS, meta_batch)]
            base_params = split_params(model)
            meta_loss = torch.tensor(0.0, device=device)
            for snr in snr_tasks:
                fp = {k: {n: p for n, p in v.items()}
                      for k, v in base_params.items()}
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
    return model, conv, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", default="fig_cl")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--trials", type=int, default=500)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[Device: {device}]", flush=True)
    train_items, test_items = load_agnews_labeled()
    bert = Extractor("bert-base-uncased", device)
    d_bert = bert.embed_dim
    cache = SplitCache(bert, train_items, test_items)
    del bert
    torch.cuda.empty_cache()

    probe, acc_clean = train_probe(cache, device)
    R = {"probe_clean_acc": acc_clean}
    PROBE_SNRS = list(EVAL_SNRS)
    conv_rows = []

    configs = [
        ("mamlB_U1_K1", dict(U=1, K=1)),
        ("mamlP_U1_K4", dict(U=1, K=4)),
        ("mamlP_U2_K4", dict(U=2, K=4)),
        ("mamlP_U3_K4", dict(U=3, K=4)),
        ("mamlP_U5_K4", dict(U=5, K=4)),
        ("mamlP_U6_K4", dict(U=6, K=4)),
        ("mamlR_U4_K4", dict(U=4, K=4, freeze_masks=True)),
    ]
    masks_store = {}
    for name, kw in configs:
        print(f"\n=== {name} ===", flush=True)
        U, K = kw.pop("U"), kw.pop("K")
        model, conv, ttime = train_maml(
            cache, U, d_bert, K, device,
            epochs=args.epochs, steps=args.steps, label=name, **kw)
        collect = PROBE_SNRS if name in (
            "mamlB_U1_K1", "mamlR_U4_K4") else None
        sweep, collected = final_eval(model, cache, U, device,
                                      trials=args.trials,
                                      collect_at=collect)
        entry = {"U": U, "K": K, "train_s": ttime,
                 "snr": {str(s): sweep[s] for s in EVAL_SNRS}}
        if collect:
            entry["probe_acc"] = {str(s): a for s, a in
                                  probe_accuracy(probe, collected,
                                                 cache, device).items()}
        if name in ("mamlP_U2_K4", "mamlP_U3_K4"):
            m = model.user_mask.weight.detach().cpu()
            mn = F.normalize(m, p=2, dim=1)
            masks_store[name] = (mn @ mn.T).numpy().tolist()
        R[name] = entry
        for c in conv:
            conv_rows.append([name, c["epoch"], c["loss"], c["cos10"]])
        R["mask_corr"] = masks_store
        with open(os.path.join(args.save_dir,
                               "cl_results_maml.json"), "w") as f:
            json.dump(R, f, indent=1)

    # probe accuracies for the already-trained maml_U4_K4 are collected
    # by re-training? No — retrain U=4 MAML for probe collection and
    # mask correlation so every reported number comes from one protocol.
    print("\n=== mamlP_U4_K4 (retrain for probe/masks) ===", flush=True)
    model, conv, ttime = train_maml(cache, 4, d_bert, 4, device,
                                    epochs=args.epochs, steps=args.steps,
                                    label="mamlP_U4_K4")
    sweep, collected = final_eval(model, cache, 4, device,
                                  trials=args.trials,
                                  collect_at=PROBE_SNRS)
    entry = {"U": 4, "K": 4, "train_s": ttime,
             "snr": {str(s): sweep[s] for s in EVAL_SNRS}}
    entry["probe_acc"] = {str(s): a for s, a in
                          probe_accuracy(probe, collected,
                                         cache, device).items()}
    m = model.user_mask.weight.detach().cpu()
    mn = F.normalize(m, p=2, dim=1)
    masks_store["mamlP_U4_K4"] = (mn @ mn.T).numpy().tolist()
    ms_gpu, n_params = measure_latency(model, cache, 4, device)
    entry["lat_gpu_ms"] = ms_gpu
    entry["params"] = n_params
    R["mamlP_U4_K4"] = entry
    R["mask_corr"] = masks_store
    for c in conv:
        conv_rows.append(["mamlP_U4_K4", c["epoch"], c["loss"],
                          c["cos10"]])

    with open(os.path.join(args.save_dir, "cl_results_maml.json"),
              "w") as f:
        json.dump(R, f, indent=1)
    with open(os.path.join(args.save_dir, "cl_convergence_maml.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "epoch", "loss", "cos10_test"])
        w.writerows(conv_rows)
    print("\nAll MAML-default experiments complete.", flush=True)


if __name__ == "__main__":
    main()
