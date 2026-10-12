# =========================================================
# rev_lib.py — shared library for the CL2026-2366 revision
#
# Revised system model (reviewer comments R1.3, R1.4):
#   * per-user power normalization: every user scales its own masked
#     signal to the energy d_s/U per block, so the total block energy is
#     d_s for every scheme and every U (no joint normalization);
#   * fading is either common (one |h| per block, as in the submission)
#     or independent (one |h_u| per user, phase pre-compensated), with
#     an optional residual phase error and per-user SNR offsets.
#
# Schemes:
#   shared : expanded shared block d_s = K d_b, learnable masks,
#            user-wise attention receiver (the proposal)
#   ortho  : orthogonal dimension sharing, user u owns d_s//U exclusive
#            dimensions (expansion for U<K, compression for U>K)
#
# Training: first-order MAML over SNR tasks, batched over frames.
# Evaluation: one inner step per SNR, then 2,000 held-out frames.
# =========================================================
import math, os, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
EVAL_SNRS = (0, 5, 10, 15, 20, 25, 30)
TRAIN_SNRS = (0, 5, 10, 15, 20, 25)
KAPPA = 8.0          # attention temperature (manuscript eq. 6)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ---------------------------------------------------------
# Data: labeled sentences, disjoint split, cached embeddings
# ---------------------------------------------------------
def load_items(dataset, n_train=8000, n_test=2000, min_len=5, max_len=30,
               seed=42):
    from datasets import load_dataset
    items = []
    if dataset == "ag_news":
        try:
            ds = load_dataset("ag_news", split="train")
        except Exception:
            ds = load_dataset("fancyzhx/ag_news", split="train")
        for ex in ds:
            first = ex["text"].split(".")[0].strip()
            if min_len <= len(first.split()) <= max_len:
                items.append((first, int(ex["label"])))
            if len(items) >= n_train + n_test:
                break
    elif dataset == "sst2":
        ds = load_dataset("stanfordnlp/sst2", split="train")
        for ex in ds:
            s = ex["sentence"].strip()
            if min_len <= len(s.split()) <= max_len:
                items.append((s, int(ex["label"])))
            if len(items) >= n_train + n_test:
                break
    else:
        raise ValueError(dataset)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n_train], items[n_train:n_train + n_test]


class Extractor:
    """Frozen encoder (BERT or DistilBERT), mean-pooled over tokens."""
    def __init__(self, model_name, device):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.embed_dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts, max_length=64, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            inp = self.tokenizer(texts[i:i + bs], padding=True,
                                 truncation=True, max_length=max_length,
                                 return_tensors="pt").to(self.device)
            h = self.model(**inp).last_hidden_state
            m = inp["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1.0)).cpu())
        return torch.cat(out, 0)


def embedding_cache(dataset, model_name, device):
    """Centered (train mean) and l2-normalized embeddings, cached on disk.
    Returns dict with train, test (N,d), train_labels, test_labels, texts."""
    tag = model_name.replace("/", "_")
    path = os.path.join(DATA, f"emb_{dataset}_{tag}.pt")
    if os.path.exists(path):
        return torch.load(path)
    os.makedirs(DATA, exist_ok=True)
    tr, te = load_items(dataset)
    ext = Extractor(model_name, device)
    E_tr = ext.encode([t for t, _ in tr])
    E_te = ext.encode([t for t, _ in te])
    mu = E_tr.mean(0, keepdim=True)
    C = {"train": F.normalize(E_tr - mu, dim=-1),
         "test": F.normalize(E_te - mu, dim=-1),
         "train_labels": torch.tensor([l for _, l in tr]),
         "test_labels": torch.tensor([l for _, l in te]),
         "train_texts": [t for t, _ in tr], "test_texts": [t for t, _ in te],
         "mu": mu, "dataset": dataset, "model": model_name,
         "n_classes": int(max(l for _, l in tr)) + 1}
    torch.save(C, path)
    return C


# ---------------------------------------------------------
# Channel: per-user gains and noise
# ---------------------------------------------------------
def user_gains(B, U, device, fading="indep", phase_std_deg=0.0,
               offsets_db=None):
    """Real-valued per-user amplitude gains g (B,U).
    fading: 'common' one Rayleigh magnitude per block, 'indep' one per
    user, 'none' unit gains. A residual phase error theta_u ~ N(0, s^2)
    after per-user phase pre-compensation leaves the factor cos(theta_u).
    offsets_db (U,) are per-user average-SNR offsets (near-far)."""
    if fading == "none":
        g = torch.ones(B, U, device=device)
    else:
        n = U if fading == "indep" else 1
        hr = torch.randn(B, n, device=device)
        hi = torch.randn(B, n, device=device)
        g = torch.sqrt((hr ** 2 + hi ** 2) / 2.0).expand(B, U)
    if phase_std_deg > 0:
        th = torch.randn(B, U, device=device) * math.radians(phase_std_deg)
        g = g * torch.cos(th)
    if offsets_db is not None:
        g = g * (10 ** (torch.as_tensor(offsets_db, device=device,
                                        dtype=torch.float32) / 20.0))
    return g


# ---------------------------------------------------------
# Transceivers (manual functional forward for MAML)
# ---------------------------------------------------------
class SharedMux(nn.Module):
    """Expanded shared block with learnable masks and user-wise attention.
    U_max masks and queries are stored; a frame may activate any subset."""
    def __init__(self, U_max, d_b, d_s, attention=True, per_user_rx=False):
        super().__init__()
        self.U_max, self.d_b, self.d_s = U_max, d_b, d_s
        self.attention = attention
        self.per_user_rx = per_user_rx
        self.tx_w = nn.Parameter(torch.empty(d_s, d_b))
        self.tx_b = nn.Parameter(torch.zeros(d_s))
        self.tx_ln_w = nn.Parameter(torch.ones(d_s))
        self.tx_ln_b = nn.Parameter(torch.zeros(d_s))
        self.mask = nn.Parameter(torch.randn(U_max, d_s) * 0.5)
        self.query = nn.Parameter(torch.randn(U_max, d_s) * 0.5)
        self.rx_ln_w = nn.Parameter(torch.ones(d_s))
        self.rx_ln_b = nn.Parameter(torch.zeros(d_s))
        if per_user_rx:        # user-specific reverse projection
            self.rx_w = nn.Parameter(torch.empty(U_max, d_b, d_s))
            self.rx_b = nn.Parameter(torch.zeros(U_max, d_b))
            for k in range(U_max):
                nn.init.kaiming_uniform_(self.rx_w.data[k], a=math.sqrt(5))
        else:
            self.rx_w = nn.Parameter(torch.empty(d_b, d_s))
            self.rx_b = nn.Parameter(torch.zeros(d_b))
            nn.init.kaiming_uniform_(self.rx_w, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.tx_w, a=math.sqrt(5))
        bound = 1 / math.sqrt(d_b)
        nn.init.uniform_(self.tx_b, -bound, bound)
        nn.init.uniform_(self.rx_b, -1 / math.sqrt(d_s), 1 / math.sqrt(d_s))

    def params(self):
        return {k: v for k, v in self.named_parameters()}

    def decode(self, z, idx, p):
        """Reverse projection, shared or user-specific."""
        if not self.per_user_rx:
            return F.linear(z, p["rx_w"], p["rx_b"])
        B, U, _ = z.shape
        zf = z.reshape(B * U, self.d_s)
        allk = torch.stack([F.linear(zf, p["rx_w"][k], p["rx_b"][k])
                            for k in range(self.U_max)], 0)   # (U_max,BU,d_b)
        sel = allk[idx.reshape(-1), torch.arange(B * U, device=z.device)]
        return sel.reshape(B, U, self.d_b)

    def forward(self, b, idx, snr_db, gains, p=None, return_mid=False):
        """b (B,U,d_b) embeddings of the active users, idx (B,U) long
        user indices, gains (B,U) channel amplitudes, snr_db scalar.
        Returns b_hat (B,U,d_b)."""
        if p is None:
            p = self.params()
        B, U, _ = b.shape
        e = F.linear(b, p["tx_w"], p["tx_b"])
        e = F.layer_norm(e, (self.d_s,), p["tx_ln_w"], p["tx_ln_b"])
        m = p["mask"][idx]                                  # (B,U,d_s)
        x = e * m
        # per-user power normalization: energy d_s/U per user per block
        x = x * torch.sqrt(self.d_s / U / (x.pow(2).sum(-1, keepdim=True)
                                            + 1e-8))
        y = (gains.unsqueeze(-1) * x).sum(1)                # (B,d_s)
        sigma = math.sqrt(1.0 / (10 ** (snr_db / 10.0)))
        y = y + sigma * torch.randn_like(y)
        r = y.unsqueeze(1) * m                              # (B,U,d_s)
        if self.attention and U > 1:
            q = p["query"][idx]                             # (B,U,d_s)
            rn = F.normalize(r, dim=-1)
            qn = F.normalize(q, dim=-1)
            scores = KAPPA * torch.einsum("bud,bvd->buv", qn, rn)
            a = torch.softmax(scores, dim=-1)               # (B,U_tgt,U_cand)
            z = torch.einsum("buv,bvd->bud", a, r)
        else:
            a = None
            z = r
        z = F.layer_norm(z, (self.d_s,), p["rx_ln_w"], p["rx_ln_b"])
        b_hat = self.decode(z, idx, p)
        if return_mid:
            return b_hat, {"attn": a, "x": x, "y": y}
        return b_hat


class OrthoDS(nn.Module):
    """Orthogonal dimension sharing: U users, each owns d_u = d_s//U
    exclusive dimensions, shared linear encoder/decoder, no interference."""
    def __init__(self, U, d_b, d_s):
        super().__init__()
        self.U, self.d_b, self.d_s = U, d_b, d_s
        self.d_u = d_s // U
        d_u = self.d_u
        self.tx_w = nn.Parameter(torch.empty(d_u, d_b))
        self.tx_b = nn.Parameter(torch.zeros(d_u))
        self.tx_ln_w = nn.Parameter(torch.ones(d_u))
        self.tx_ln_b = nn.Parameter(torch.zeros(d_u))
        self.rx_ln_w = nn.Parameter(torch.ones(d_u))
        self.rx_ln_b = nn.Parameter(torch.zeros(d_u))
        self.rx_w = nn.Parameter(torch.empty(d_b, d_u))
        self.rx_b = nn.Parameter(torch.zeros(d_b))
        nn.init.kaiming_uniform_(self.tx_w, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.rx_w, a=math.sqrt(5))
        nn.init.uniform_(self.tx_b, -1 / math.sqrt(d_b), 1 / math.sqrt(d_b))
        nn.init.uniform_(self.rx_b, -1 / math.sqrt(d_u), 1 / math.sqrt(d_u))

    def params(self):
        return {k: v for k, v in self.named_parameters()}

    def forward(self, b, idx, snr_db, gains, p=None, return_mid=False):
        if p is None:
            p = self.params()
        B, U, _ = b.shape
        x = F.linear(b, p["tx_w"], p["tx_b"])
        x = F.layer_norm(x, (self.d_u,), p["tx_ln_w"], p["tx_ln_b"])
        # unit power per exclusive dimension: energy d_u per user per block
        x = x * torch.sqrt(self.d_u / (x.pow(2).sum(-1, keepdim=True) + 1e-8))
        sigma = math.sqrt(1.0 / (10 ** (snr_db / 10.0)))
        y = gains.unsqueeze(-1) * x + sigma * torch.randn_like(x)
        z = F.layer_norm(y, (self.d_u,), p["rx_ln_w"], p["rx_ln_b"])
        b_hat = F.linear(z, p["rx_w"], p["rx_b"])
        if return_mid:
            return b_hat, {"x": x, "y": y}
        return b_hat


def build_model(scheme, U_max, d_b, K, attention=True, per_user_rx=False):
    if scheme == "shared":
        return SharedMux(U_max, d_b, d_b * K, attention=attention,
                         per_user_rx=per_user_rx)
    if scheme == "ortho":
        return OrthoDS(U_max, d_b, d_b * K)
    raise ValueError(scheme)


# ---------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------
def semantic_loss(b, b_hat, lam=0.5):
    mse = F.mse_loss(b_hat, b)
    cos = F.cosine_similarity(b_hat, b, dim=-1).mean()
    return (1 - lam) * mse + lam * (1 - cos)


# ---------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------
def sample_frames(E, B, U, U_max, device, gen=None, agnostic=False):
    """Returns embeddings (B,U,d) and user indices (B,U).
    agnostic=True draws a random subset of U users out of U_max per frame,
    otherwise users 0..U-1 are active."""
    N = E.shape[0]
    pos = torch.randint(0, N, (B, U), generator=gen)
    b = E[pos].to(device)
    if agnostic:
        idx = torch.argsort(torch.rand(B, U_max, generator=gen), dim=1)[:, :U]
    else:
        idx = torch.arange(U).unsqueeze(0).expand(B, U)
    return b, idx.to(device), pos


# ---------------------------------------------------------
# MAML training
# ---------------------------------------------------------
def inner_step(model, p, b, idx, snr, gains, alpha, train_keys, lam):
    loss = semantic_loss(b, model(b, idx, snr, gains, p=p), lam)
    keys = [k for k in train_keys]
    grads = torch.autograd.grad(loss, [p[k] for k in keys],
                                allow_unused=True)
    q = dict(p)
    for k, g in zip(keys, grads):
        if g is not None:                       # unused (query at U=1)
            q[k] = p[k] - alpha * g.detach()    # first-order MAML
    return q, loss.item()


def train(model, cache, cfg, device, log=print):
    """cfg keys: U, U_max, agnostic, epochs, steps, batch, meta_batch,
    alpha, beta, inner_steps, lam, fading, train_snrs, mode ('maml' or
    'joint'), train_keys (list of parameter names to adapt)."""
    E = cache["train"]
    gen = torch.Generator().manual_seed(cfg["seed"])
    train_keys = cfg["train_keys"]
    opt = torch.optim.Adam([p for n, p in model.named_parameters()
                            if n in train_keys], lr=cfg["beta"])
    conv = []
    t0 = time.time()
    U_max = cfg["U_max"]
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        loss_sum = 0.0
        for _ in range(cfg["steps"]):
            U = (int(torch.randint(1, U_max + 1, (1,), generator=gen))
                 if cfg["agnostic"] else cfg["U"])
            if cfg["mode"] == "joint":
                snr = float(random.choice(cfg["train_snrs"]))
                b, idx, _ = sample_frames(E, cfg["batch"], U, U_max, device,
                                          gen, cfg["agnostic"])
                g = user_gains(cfg["batch"], U, device, cfg["fading"])
                loss = semantic_loss(b, model(b, idx, snr, g), cfg["lam"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_sum += loss.item()
                continue
            pool = list(cfg["train_snrs"])
            tasks = (random.sample(pool, cfg["meta_batch"]) if cfg["meta_batch"] <= len(pool)
                     else random.choices(pool, k=cfg["meta_batch"]))
            base = model.params()
            meta = 0.0
            for snr in tasks:
                p = dict(base)
                for _ in range(cfg["inner_steps"]):
                    b, idx, _ = sample_frames(E, cfg["batch"], U, U_max,
                                              device, gen, cfg["agnostic"])
                    g = user_gains(cfg["batch"], U, device, cfg["fading"])
                    p, _ = inner_step(model, p, b, idx, snr, g, cfg["alpha"],
                                      train_keys, cfg["lam"])
                b, idx, _ = sample_frames(E, cfg["batch"], U, U_max, device,
                                          gen, cfg["agnostic"])
                g = user_gains(cfg["batch"], U, device, cfg["fading"])
                meta = meta + semantic_loss(b, model(b, idx, snr, g, p=p),
                                            cfg["lam"])
            meta = meta / cfg["meta_batch"]
            opt.zero_grad(set_to_none=True)
            meta.backward()
            opt.step()
            loss_sum += meta.item()
        # light held-out convergence probe at 10 dB (no adaptation)
        c10 = quick_eval(model, cache, cfg, device, 10, n=256)
        conv.append((ep, loss_sum / cfg["steps"], c10))
        if ep == 1 or ep % 10 == 0 or ep == cfg["epochs"]:
            log(f"  [{cfg['name']} ep {ep:03d}] loss={loss_sum/cfg['steps']:.4f}"
                f" cos@10(test,no-adapt)={c10:.4f} t={time.time()-t0:.0f}s")
    return conv, time.time() - t0


@torch.no_grad()
def quick_eval(model, cache, cfg, device, snr, n=256, p=None, U=None,
               fading=None, gen=None):
    model.eval()
    U = U or cfg["U"]
    b, idx, _ = sample_frames(cache["test"], n, U, cfg["U_max"], device,
                              gen, cfg["agnostic"])
    g = user_gains(n, U, device, fading or cfg["fading"])
    bh = model(b, idx, snr, g, p=p)
    return F.cosine_similarity(bh, b, dim=-1).mean().item()


# ---------------------------------------------------------
# Adaptation and held-out evaluation
# ---------------------------------------------------------
def adapt(model, cache, cfg, device, snr, U, steps=None, n=256):
    """One (or cfg['adapt_steps']) inner step(s) at the target SNR on
    training frames, returning the adapted parameter dict."""
    steps = cfg.get("adapt_steps", 1) if steps is None else steps
    if steps == 0:
        return None
    gen = torch.Generator().manual_seed(cfg["seed"] + 1000 + snr)
    p = dict(model.params())
    model.train()
    with torch.enable_grad():
        for _ in range(steps):
            b, idx, _ = sample_frames(cache["train"], n, U, cfg["U_max"],
                                      device, gen, cfg["agnostic"])
            g = user_gains(n, U, device, cfg["fading"])
            p, _ = inner_step(model, p, b, idx, snr, g, cfg["alpha"],
                              cfg["train_keys"], cfg["lam"])
    return {k: v.detach() for k, v in p.items()}


@torch.no_grad()
def evaluate(model, cache, cfg, device, U, trials=2000, bs=250,
             fading=None, phase_std_deg=0.0, offsets_db=None,
             adapt_steps=None, probe=None, eval_seed=123,
             snrs=EVAL_SNRS):
    """Held-out SNR sweep. Returns dict snr -> metrics (mean CosSim, MSE,
    per-user CosSim, std over frames, probe accuracy if probe given)."""
    out = {}
    E = cache["test"]
    for snr in snrs:
        p = adapt(model, cache, cfg, device, snr, U, steps=adapt_steps)
        set_seed(eval_seed + snr)
        gen = torch.Generator().manual_seed(eval_seed + snr)
        model.eval()
        cos_all, mse_all, pos_all, bh_all = [], [], [], []
        for _ in range(trials // bs):
            b, idx, pos = sample_frames(E, bs, U, cfg["U_max"], device, gen,
                                        cfg["agnostic"])
            g = user_gains(bs, U, device, fading or cfg["fading"],
                           phase_std_deg, offsets_db)
            bh = model(b, idx, snr, g, p=p)
            cos_all.append(F.cosine_similarity(bh, b, dim=-1).cpu())
            mse_all.append((bh - b).pow(2).mean(-1).cpu())
            if probe is not None:
                pos_all.append(pos)
                bh_all.append(bh.cpu())
        cos = torch.cat(cos_all, 0)                       # (N,U)
        mse = torch.cat(mse_all, 0)
        frame_cos = cos.mean(1)
        rec = {"cos": cos.mean().item(), "mse": mse.mean().item(),
               "cos_user": cos.mean(0).tolist(),
               "cos_worst_user": cos.mean(0).min().item(),
               "cos_frame_std": frame_cos.std().item(),
               "n_frames": int(cos.shape[0])}
        if probe is not None:
            W, _ = probe
            lab = cache["test_labels"][torch.cat(pos_all, 0)]
            pred = W(torch.cat(bh_all, 0).to(device)).argmax(-1).cpu()
            rec["probe_acc"] = (pred == lab).float().mean().item()
        out[snr] = rec
    return out


# ---------------------------------------------------------
# Linear probe on clean training embeddings
# ---------------------------------------------------------
def train_probe(cache, device, epochs=300, lr=1e-2, seed=0):
    set_seed(seed)
    X = cache["train"].to(device)
    y = cache["train_labels"].to(device)
    W = nn.Linear(X.shape[1], cache["n_classes"]).to(device)
    opt = torch.optim.Adam(W.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(W(X), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (W(cache["test"].to(device)).argmax(-1).cpu()
               == cache["test_labels"]).float().mean().item()
    return W, acc


# ---------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------
@torch.no_grad()
def mask_stats(model):
    if not hasattr(model, "mask"):
        return None
    m = F.normalize(model.mask.detach().cpu(), dim=-1)
    C = (m @ m.T).numpy()
    off = np.abs(C - np.eye(len(C)))
    # support overlap: cosine between the squared masks, 1/3 for independent
    # Gaussian masks, 0 for disjoint supports, 1 for identical supports
    m2 = F.normalize(model.mask.detach().cpu() ** 2, dim=-1)
    O = (m2 @ m2.T).numpy()
    offO = O - np.eye(len(O))
    n = len(C)
    return {"max_abs_offdiag": float(off.max()),
            "mean_abs_offdiag": float(off.sum() / (n * (n - 1))) if n > 1 else 0.0,
            "support_overlap_mean": float(offO.sum() / (n * (n - 1))) if n > 1 else 0.0,
            "support_overlap_max": float(offO.max()) if n > 1 else 0.0}


def effective_rank(E):
    """Participation ratio and 90/99 percent variance dimensions."""
    X = E - E.mean(0, keepdim=True)
    s = torch.linalg.svdvals(X.double())
    lam = (s ** 2) / (X.shape[0] - 1)
    pr = (lam.sum() ** 2 / (lam ** 2).sum()).item()
    c = torch.cumsum(lam, 0) / lam.sum()
    return {"participation_ratio": pr,
            "dims_90": int((c < 0.90).sum().item()) + 1,
            "dims_99": int((c < 0.99).sum().item()) + 1}


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def measure_latency(model, cache, cfg, device, U, n=200):
    b, idx, _ = sample_frames(cache["test"], 1, U, cfg["U_max"], device)
    g = user_gains(1, U, device, cfg["fading"])
    for _ in range(20):
        model(b, idx, 10, g)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(n):
        model(b, idx, 10, g)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    return 1000.0 * (time.time() - t0) / n
