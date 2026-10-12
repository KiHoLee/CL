# =========================================================
# rev_run.py — train and evaluate one configuration of the revision
# protocol and write data/<name>_s<seed>.json. See rev_queue.sh for the
# list of configurations behind every figure and table.
# =========================================================
import argparse, json, os, time, sys
import torch
import torch.nn.functional as F

from rev_lib import (DATA, EVAL_SNRS, TRAIN_SNRS, get_device, set_seed,
                     embedding_cache, build_model, train, evaluate,
                     train_probe, mask_stats, effective_rank, count_params,
                     measure_latency)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--scheme", choices=["shared", "ortho"], default="shared")
    ap.add_argument("--U", type=int, default=4)
    ap.add_argument("--U-max", type=int, default=None,
                    help="number of stored masks (agnostic training)")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fading", choices=["indep", "common", "none"],
                    default="indep")
    ap.add_argument("--agnostic", action="store_true",
                    help="train one model over U=1..U_max with random "
                         "active subsets")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--meta-batch", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=5e-4)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--inner-steps", type=int, default=1)
    ap.add_argument("--adapt-steps", type=int, default=1)
    ap.add_argument("--mode", choices=["maml", "joint"], default="maml")
    ap.add_argument("--train-snrs", type=float, nargs="+",
                    default=list(TRAIN_SNRS))
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--freeze", default="",
                    help="comma list among mask,tx (kept at random init)")
    ap.add_argument("--no-attention", action="store_true")
    ap.add_argument("--per-user-rx", action="store_true",
                    help="user-specific reverse projection at the receiver")
    ap.add_argument("--dataset", default="ag_news")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--eval-U", type=int, nargs="*", default=None,
                    help="user counts to evaluate (agnostic model)")
    ap.add_argument("--robust", action="store_true",
                    help="extra evaluations: phase error, near-far "
                         "offsets, fading mismatch, no adaptation")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--out-dir", default=DATA)
    args = ap.parse_args()

    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.name}_s{args.seed}.json")
    log_lines = []

    def log(s):
        print(s, flush=True)
        log_lines.append(s)

    log(f"[{args.name} seed {args.seed}] device={device}")
    cache = embedding_cache(args.dataset, args.model, device)
    d_b = cache["train"].shape[1]
    U_max = args.U_max or args.U
    set_seed(args.seed)
    model = build_model(args.scheme, U_max, d_b, args.K,
                        attention=not args.no_attention,
                        per_user_rx=args.per_user_rx).to(device)
    frozen = set()
    if "mask" in args.freeze:
        frozen.add("mask")
    if "tx" in args.freeze:
        frozen.update({"tx_w", "tx_b", "tx_ln_w", "tx_ln_b"})
    train_keys = [n for n, _ in model.named_parameters() if n not in frozen]
    cfg = dict(name=args.name, seed=args.seed, scheme=args.scheme, U=args.U,
               U_max=U_max, K=args.K, d_b=d_b, d_s=d_b * args.K,
               agnostic=args.agnostic, epochs=args.epochs, steps=args.steps,
               batch=args.batch, meta_batch=args.meta_batch,
               alpha=args.alpha, beta=args.beta, inner_steps=args.inner_steps,
               adapt_steps=args.adapt_steps, mode=args.mode,
               train_snrs=list(args.train_snrs), lam=args.lam,
               fading=args.fading, freeze=sorted(frozen),
               attention=not args.no_attention, per_user_rx=args.per_user_rx,
               dataset=args.dataset,
               model=args.model, trials=args.trials, train_keys=train_keys,
               device=str(device), torch=torch.__version__)
    log(f"  cfg: {json.dumps({k: v for k, v in cfg.items() if k != 'train_keys'})}")
    log(f"  params={count_params(model)} trainable={sum(p.numel() for n,p in model.named_parameters() if n in train_keys)}")

    conv, ttime = train(model, cache, cfg, device, log=log)
    R = {"cfg": cfg, "train_s": ttime, "params": count_params(model),
         "conv": conv, "mask_stats": mask_stats(model)}

    probe = None
    if args.probe:
        probe = train_probe(cache, device)
        R["probe_clean_acc"] = probe[1]
        log(f"  probe clean acc = {probe[1]:.4f}")

    eval_Us = args.eval_U or [args.U]
    R["eval"] = {}
    for U in eval_Us:
        res = evaluate(model, cache, cfg, device, U, trials=args.trials,
                       probe=probe)
        R["eval"][str(U)] = {str(s): res[s] for s in EVAL_SNRS}
        log(f"  U={U}: " + " ".join(f"{s}:{res[s]['cos']:.4f}" for s in EVAL_SNRS)
            + (" | probe " + " ".join(f"{s}:{res[s]['probe_acc']:.3f}" for s in EVAL_SNRS)
               if probe else ""))

    if args.robust:
        U = args.U
        R["robust"] = {}
        # (a) no adaptation step (meta-initialization used directly)
        res = evaluate(model, cache, cfg, device, U, trials=args.trials,
                       adapt_steps=0)
        R["robust"]["no_adapt"] = {str(s): res[s]["cos"] for s in EVAL_SNRS}
        # (b) residual phase error after per-user pre-compensation
        for deg in (10, 20, 30):
            res = evaluate(model, cache, cfg, device, U, trials=args.trials,
                           phase_std_deg=deg)
            R["robust"][f"phase{deg}"] = {str(s): res[s]["cos"] for s in EVAL_SNRS}
        # (c) near-far: per-user average-SNR offsets, mean and worst user
        for D in (3, 6):
            off = [(-D + 2 * D * u / max(U - 1, 1)) for u in range(U)]
            res = evaluate(model, cache, cfg, device, U, trials=args.trials,
                           offsets_db=off)
            R["robust"][f"nearfar{D}"] = {
                str(s): {"cos": res[s]["cos"],
                         "worst": res[s]["cos_worst_user"],
                         "cos_user": res[s]["cos_user"]} for s in EVAL_SNRS}
        # (d) fading mismatch: model trained under cfg fading, tested under
        #     the other model
        other = "common" if args.fading == "indep" else "indep"
        res = evaluate(model, cache, cfg, device, U, trials=args.trials,
                       fading=other)
        R["robust"][f"test_{other}"] = {str(s): res[s]["cos"] for s in EVAL_SNRS}
        log("  robust: " + json.dumps({k: (v.get("10") if isinstance(v.get("10"), float) else v.get("10", {}).get("cos")) for k, v in R["robust"].items()}))

    if args.latency:
        R["latency_ms"] = measure_latency(model, cache, cfg, device, args.U)
        log(f"  latency {R['latency_ms']:.3f} ms/frame")

    if args.scheme == "shared" and args.seed == 1 and args.U == 4 and args.K == 4 \
            and not args.freeze and not args.agnostic:
        R["effective_rank_test"] = effective_rank(cache["test"])

    R["log"] = log_lines
    with open(out_path, "w") as f:
        json.dump(R, f, indent=1)
    ck = os.path.join(args.out_dir, "ckpt")
    os.makedirs(ck, exist_ok=True)
    torch.save({"cfg": cfg, "state_dict": {k: v.cpu() for k, v in
                                           model.state_dict().items()}},
               os.path.join(ck, f"{args.name}_s{args.seed}.pt"))
    log(f"  wrote {out_path} (train {ttime/60:.1f} min)")


if __name__ == "__main__":
    main()
