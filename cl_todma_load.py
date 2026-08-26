# cl_todma_load.py — ToDMA load sweep (evaluation only, no training).
#
# Evaluates the ToDMA token-domain scheme (T=24, L=128, genie-aided
# association) for U in {1,2,3,5,6} on the held-out test set, matching
# the U=4 run stored in cl_results.json, so that Fig. 3 can show the
# ToDMA aggregate fidelity across load.
import argparse, os, json
import torch

from cl_experiments import (
    load_agnews_labeled, Extractor, SplitCache, todma_eval, EVAL_SNRS
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", default="fig_cl")
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[Device: {device}]", flush=True)
    train_items, test_items = load_agnews_labeled()
    bert = Extractor("bert-base-uncased", device)
    cache = SplitCache(bert, train_items, test_items)

    R = {}
    for U in [1, 2, 3, 5, 6]:
        print(f"\n=== ToDMA U={U} (T=24, L=128) ===", flush=True)
        res, _ = todma_eval(bert, cache, device, U=U, T=24, L=128,
                            n_frames=args.frames)
        key = f"todma_U{U}_T24_L128"
        R[key] = {str(s): res[s] for s in EVAL_SNRS}
        with open(os.path.join(args.save_dir,
                               "cl_results_todma_u.json"), "w") as f:
            json.dump(R, f, indent=1)
    print("\nToDMA load sweep complete.", flush=True)


if __name__ == "__main__":
    main()
