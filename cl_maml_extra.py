# cl_maml_extra.py — MAML-trained K sweep (K=1,2,8) and DistilBERT
# replication, completing the unified MAML protocol.
import argparse, os, json, csv
import torch
import torch.nn.functional as F

from cl_experiments import (
    load_agnews_labeled, Extractor, SplitCache, final_eval, set_seed,
    EVAL_SNRS
)
from cl_maml_all import train_maml


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

    R = {}
    conv_rows = []
    for name, K in [("mamlK_U4_K1", 1), ("mamlK_U4_K2", 2),
                    ("mamlK_U4_K8", 8)]:
        print(f"\n=== {name} ===", flush=True)
        model, conv, ttime = train_maml(cache, 4, d_bert, K, device,
                                        epochs=args.epochs,
                                        steps=args.steps, label=name)
        sweep, _ = final_eval(model, cache, 4, device, trials=args.trials)
        R[name] = {"U": 4, "K": K, "train_s": ttime,
                   "snr": {str(s): sweep[s] for s in EVAL_SNRS}}
        for c in conv:
            conv_rows.append([name, c["epoch"], c["loss"], c["cos10"]])
        with open(os.path.join(args.save_dir,
                               "cl_results_maml2.json"), "w") as f:
            json.dump(R, f, indent=1)

    print("\n=== mamlD_U4_K4 (DistilBERT) ===", flush=True)
    distil = Extractor("distilbert-base-uncased", device)
    dcache = SplitCache(distil, train_items, test_items)
    d_d = distil.embed_dim
    del distil
    torch.cuda.empty_cache()
    model, conv, ttime = train_maml(dcache, 4, d_d, 4, device,
                                    epochs=args.epochs,
                                    steps=args.steps, label="mamlD")
    sweep, _ = final_eval(model, dcache, 4, device, trials=args.trials)
    R["mamlD_U4_K4"] = {"U": 4, "K": 4, "train_s": ttime,
                        "snr": {str(s): sweep[s] for s in EVAL_SNRS}}
    for c in conv:
        conv_rows.append(["mamlD_U4_K4", c["epoch"], c["loss"],
                          c["cos10"]])

    with open(os.path.join(args.save_dir, "cl_results_maml2.json"),
              "w") as f:
        json.dump(R, f, indent=1)
    with open(os.path.join(args.save_dir, "cl_convergence_maml2.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "epoch", "loss", "cos10_test"])
        w.writerows(conv_rows)
    print("\nExtra MAML experiments complete.", flush=True)


if __name__ == "__main__":
    main()
