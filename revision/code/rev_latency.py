# =========================================================
# rev_latency.py — per-frame inference latency and parameter counts of
# the trained proposal and ODS models on ONE device, from the saved
# checkpoints (data/ckpt/*.pt). Writes data/rev_latency.json.
# Usage: python rev_latency.py shared_U4_s1 ortho_U4_s1 [more names]
# =========================================================
import json, os, sys, time
import torch

from rev_lib import DATA, get_device, embedding_cache, build_model, sample_frames, user_gains, count_params


def main():
    dev = get_device()
    out = {"device": str(dev), "torch": torch.__version__}
    for name in sys.argv[1:]:
        ck = torch.load(os.path.join(DATA, "ckpt", f"{name}.pt"), map_location="cpu")
        cfg = ck["cfg"]
        cache = embedding_cache(cfg["dataset"], cfg["model"], dev)
        model = build_model(cfg["scheme"], cfg["U_max"], cfg["d_b"], cfg["K"],
                            attention=cfg["attention"], per_user_rx=cfg.get("per_user_rx", False)).to(dev)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        U = cfg["U"]
        b, idx, _ = sample_frames(cache["test"], 1, U, cfg["U_max"], dev)
        g = user_gains(1, U, dev, cfg["fading"])
        with torch.no_grad():
            for _ in range(30):
                model(b, idx, 10, g)
            sync = (torch.cuda.synchronize if dev.type == "cuda" else
                    torch.mps.synchronize if dev.type == "mps" else (lambda: None))
            sync()
            t0 = time.time()
            for _ in range(300):
                model(b, idx, 10, g)
            sync()
            ms = 1000 * (time.time() - t0) / 300
        rx = 0
        for k, v in model.state_dict().items():
            if k.startswith("rx_"):
                rx += v.numel()
        out[name] = {"ms_per_frame": ms, "params": count_params(model), "params_rx": rx, "U": U}
        print(name, f"{ms:.3f} ms/frame, params {count_params(model)} (rx {rx})")
    with open(os.path.join(DATA, "rev_latency.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
