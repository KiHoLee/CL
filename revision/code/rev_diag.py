# =========================================================
# rev_diag.py — mechanism diagnostics of a trained shared-block model
# (reviewer comments R1.5 / R2.4): where does the separation come from?
#
# For a checkpoint data/ckpt/<name>_s<seed>.pt it reports, on held-out
# frames at a given SNR,
#   * mask statistics (cosine between masks, support overlap),
#   * the signal-to-interference ratio (SIR) of the receiver candidate
#     r_u = y * m_u before the attention and the linear decoder,
#   * the SIR of the demultiplexed vector z_u after attention,
#   * the attention weight on the own candidate,
#   * the CosSim obtained when the learned masks are replaced by fresh
#     random masks at test time (receiver unchanged), and when the
#     attention is bypassed (z_u = r_u),
#   * the leakage of one user's signal into another's decoded embedding
#     through the linear decoder (interference energy after f_rx relative
#     to the signal energy after f_rx).
# Usage: python rev_diag.py <ckpt> [--snr 10] [--frames 1000]
# =========================================================
import argparse, json, math, os
import torch
import torch.nn.functional as F

from rev_lib import (DATA, get_device, set_seed, embedding_cache, build_model,
                     sample_frames, user_gains, mask_stats)


@torch.no_grad()
def components(model, p, b, idx, gains, snr_db):
    """Per-user signal and interference parts of r_u and z_u."""
    B, U, _ = b.shape
    e = F.linear(b, p["tx_w"], p["tx_b"])
    e = F.layer_norm(e, (model.d_s,), p["tx_ln_w"], p["tx_ln_b"])
    m = p["mask"][idx]
    x = e * m
    x = x * torch.sqrt(model.d_s / U / (x.pow(2).sum(-1, keepdim=True) + 1e-8))
    xg = gains.unsqueeze(-1) * x                                # (B,U,d)
    sigma = math.sqrt(1.0 / (10 ** (snr_db / 10.0)))
    n = sigma * torch.randn(B, model.d_s, device=b.device)
    y = xg.sum(1) + n
    # candidate r_u = y*m_u split into own, other, noise parts
    own = xg * m                                                 # (B,U,d)
    tot = y.unsqueeze(1) * m
    other = (xg.sum(1, keepdim=True) - xg) * m
    noise = n.unsqueeze(1) * m
    out = {"sir_r": (own.pow(2).sum(-1) / other.pow(2).sum(-1)).mean().item(),
           "snr_r": (own.pow(2).sum(-1) / noise.pow(2).sum(-1)).mean().item()}
    q = p["query"][idx]
    rn, qn = F.normalize(tot, dim=-1), F.normalize(q, dim=-1)
    a = torch.softmax(8.0 * torch.einsum("bud,bvd->buv", qn, rn), dim=-1)
    out["attn_own"] = a.diagonal(dim1=1, dim2=2).mean().item()
    # z_u split: own part through attention weights
    z_own = torch.einsum("buv,bvd->bud", a, own)
    z_oth = torch.einsum("buv,bvd->bud", a, other)
    out["sir_z"] = (z_own.pow(2).sum(-1) / z_oth.pow(2).sum(-1)).mean().item()
    # leakage through the linear decoder (layer norm shifts both parts
    # equally, so the linear map alone is applied)
    d_own = model.decode(z_own, idx, {"rx_w": p["rx_w"], "rx_b": torch.zeros_like(p["rx_b"])})
    d_oth = model.decode(z_oth, idx, {"rx_w": p["rx_w"], "rx_b": torch.zeros_like(p["rx_b"])})
    out["sir_after_rx"] = (d_own.pow(2).sum(-1) / d_oth.pow(2).sum(-1)).mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--snr", type=float, default=10)
    ap.add_argument("--frames", type=int, default=1000)
    args = ap.parse_args()
    dev = get_device()
    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck["cfg"]
    cache = embedding_cache(cfg["dataset"], cfg["model"], dev)
    model = build_model(cfg["scheme"], cfg["U_max"], cfg["d_b"], cfg["K"],
                        attention=cfg["attention"],
                        per_user_rx=cfg.get("per_user_rx", False)).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    p = model.params()
    U = cfg["U"]
    set_seed(123)
    gen = torch.Generator().manual_seed(123)
    b, idx, _ = sample_frames(cache["test"], args.frames, U, cfg["U_max"], dev,
                              gen, cfg["agnostic"])
    g = user_gains(args.frames, U, dev, cfg["fading"])
    R = {"name": cfg["name"], "seed": cfg["seed"], "snr": args.snr,
         "mask_stats": mask_stats(model)}
    R.update({k: v for k, v in components(model, p, b, idx, g, args.snr).items()})
    base = F.cosine_similarity(model(b, idx, args.snr, g), b, dim=-1).mean().item()
    R["cos"] = base
    # swap learned masks for random masks at test time (receiver unchanged)
    q = dict(p)
    torch.manual_seed(7)
    q["mask"] = torch.randn_like(p["mask"]) * p["mask"].std()
    R["cos_random_masks_at_test"] = F.cosine_similarity(
        model(b, idx, args.snr, g, p=q), b, dim=-1).mean().item()
    # bypass attention
    model.attention = False
    R["cos_no_attention_at_test"] = F.cosine_similarity(
        model(b, idx, args.snr, g), b, dim=-1).mean().item()
    model.attention = cfg["attention"]
    for k, v in R.items():
        if isinstance(v, float):
            R[k] = round(v, 5)
    print(json.dumps(R, indent=1))
    out = os.path.join(DATA, "diag")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{cfg['name']}_s{cfg['seed']}_snr{int(args.snr)}.json"), "w") as f:
        json.dump(R, f, indent=1)


if __name__ == "__main__":
    main()
