# =========================================================
# bert_semcom.py
# BERT-based Multi-User Semantic Communication
# with Expanded Shared Embedding + User-Wise Attention
#
# Key Idea:
#   Each user's BERT embedding (768-dim) is EXPANDED to a
#   larger shared embedding space (e.g., 768*K where K=4),
#   enabling statistical multiplexing gain through masking
#   in the over-provisioned shared space.
#
# Architecture:
#   Tx:
#     b_u = BERT(text_u)                    # (d_bert,)  e.g., 768
#     e_u = W_tx(b_u)                       # (d_shared,) e.g., 3072
#     x_u = e_u ⊙ m_u                      # user-masked in expanded space
#     y   = Σ_u x_u                         # over-the-air superposition
#     y_rx = channel(y)
#
#   Rx (User-wise attention demux):
#     R_i = y_rx ⊙ m_i                     # candidate projections
#     a_{u,i} = softmax( <norm(R_i), norm(q_u)> / sqrt(d) )
#     z_u = Σ_i a_{u,i} R_i
#     b̂_u = W_rx(z_u)                      # (d_bert,) recovered
#
#   Loss:
#     L = (1-λ) * MSE(b_u, b̂_u) + λ * (1 - CosSim(b_u, b̂_u))
#
#   Metrics: Cosine similarity, MSE
#
# Statistical Multiplexing Gain:
#   Even with U=2 users, projecting to d_shared = d_bert*4
#   provides over-provisioned subspace for each user's mask,
#   reducing inter-user interference and improving separation.
#
# Train modes: joint / maml_decoder / maml_full
# =========================================================

import argparse
import math
import random
import csv
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.func import functional_call
except Exception:
    from torch.nn.utils.stateless import functional_call


# =========================================================
# BERT Embedding Extractor (frozen)
# =========================================================
class BertEmbeddingExtractor:
    """
    Extract sentence-level embeddings from pre-trained BERT.
    Uses mean pooling over non-padding tokens to mitigate
    BERT [CLS] anisotropy. Returns un-normalized embeddings;
    centering and L2-norm are applied at the cache level.
    """
    def __init__(self, model_name="bert-base-uncased", device="cpu"):
        from transformers import BertModel, BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.embed_dim = self.model.config.hidden_size  # 768

    @torch.no_grad()
    def encode(self, texts):
        """Mean-pooled token embedding (excluding padding)."""
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=64, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state          # (B, T, d)
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
        summed = (last_hidden * mask).sum(dim=1)         # (B, d)
        count = mask.sum(dim=1).clamp(min=1.0)           # (B, 1)
        mean_emb = summed / count                        # (B, d)
        return mean_emb


# =========================================================
# Text Dataset
# =========================================================
def load_sentences(source="ag_news", max_sentences=50000,
                   min_len=5, max_len=30):
    """Load sentences from various sources."""
    sentences = []

    if source == "europarl":
        try:
            from datasets import load_dataset
            ds = load_dataset("wmt14", "de-en", split="train", streaming=True)
            for example in ds:
                sent = example["translation"]["en"]
                words = sent.split()
                if min_len <= len(words) <= max_len:
                    sentences.append(sent)
                if len(sentences) >= max_sentences:
                    break
        except Exception:
            print("[INFO] WMT14 not available, trying ag_news...")
            source = "ag_news"

    if source == "ag_news":
        try:
            from datasets import load_dataset
            ds = load_dataset("ag_news", split="train")
            for example in ds:
                sent = example["text"]
                first_sent = sent.split(".")[0].strip()
                words = first_sent.split()
                if min_len <= len(words) <= max_len:
                    sentences.append(first_sent)
                if len(sentences) >= max_sentences:
                    break
        except Exception:
            print("[INFO] ag_news not available, using synthetic...")
            source = "synthetic"

    if source == "synthetic" or len(sentences) < 1000:
        print("[INFO] Generating synthetic diverse sentences...")
        templates = [
            "The {} {} the {} in the {}.",
            "A {} {} quickly {} the {}.",
            "Several {} {} near the {} {}.",
            "The {} and {} {} {} together.",
            "Every {} must {} its own {}.",
            "Under the {}, a {} {} {} softly.",
            "The {} of {} {} a {} signal.",
            "Without {}, the {} cannot {} properly.",
            "A new {} {} from the {} {}.",
            "The {} {} through the {} channel.",
        ]
        nouns = ["system", "signal", "network", "channel", "user", "device",
                 "antenna", "receiver", "transmitter", "waveform", "protocol",
                 "data", "packet", "frequency", "power", "noise", "beam",
                 "satellite", "tower", "base station", "terminal", "sensor",
                 "message", "code", "sequence", "spectrum", "bandwidth"]
        verbs = ["processes", "transmits", "receives", "analyzes", "encodes",
                 "decodes", "modulates", "filters", "amplifies", "detects",
                 "estimates", "optimizes", "adapts", "allocates", "schedules"]
        adjs = ["wireless", "digital", "analog", "robust", "adaptive",
                "cognitive", "massive", "distributed", "cooperative", "mobile",
                "reliable", "efficient", "dynamic", "intelligent", "semantic"]
        for _ in range(max(max_sentences, 50000)):
            tmpl = random.choice(templates)
            n_slots = tmpl.count("{}")
            fillers = [random.choice(random.choice([nouns, verbs, adjs]))
                       for _ in range(n_slots)]
            sentences.append(tmpl.format(*fillers))

    random.shuffle(sentences)
    return sentences[:max_sentences]


# =========================================================
# Channel
# =========================================================
def apply_channel(y, snr_db, channel="rayleigh"):
    """
    y: (d_shared,) real-valued, power-normalized embedding vector.
    Rayleigh: magnitude fading |h| where h ~ CN(0,1).
    """
    snr_lin = 10 ** (snr_db / 10.0)
    noise_var = 1.0 / snr_lin

    if channel == "rayleigh":
        # Complex Rayleigh fading -> magnitude |h|
        h_real = torch.randn((), device=y.device)
        h_imag = torch.randn((), device=y.device)
        h_mag = torch.sqrt(h_real**2 + h_imag**2) / math.sqrt(2.0)
        y = h_mag * y

    noise = torch.randn_like(y) * math.sqrt(noise_var)
    return y + noise


# =========================================================
# Model: Expanded Shared Embedding + User-Wise Attention
# =========================================================
class BertSemComMux(nn.Module):
    """
    BERT-based Multi-User Semantic Communication.

    Key design: BERT embedding (d_bert=768) is EXPANDED to
    d_shared = d_bert * mux_factor (e.g., 768*4=3072) to
    provide statistical multiplexing gain.

    Even with fewer users than mux_factor, the expanded space
    gives each user more "room" for orthogonal mask allocation,
    reducing inter-user interference.
    """
    def __init__(self, U, d_bert, d_shared, hidden=512):
        super().__init__()
        self.U = U
        self.d_bert = d_bert          # 768
        self.d_shared = d_shared      # e.g., 768*4 = 3072

        # Tx: single-layer linear projection to the shared space
        # (no sub-d_bert bottleneck, no costly d_shared x d_shared layer).
        # Followed by LayerNorm for numerical stability.
        self.tx_proj = nn.Sequential(
            nn.Linear(d_bert, d_shared),
            nn.LayerNorm(d_shared),
        )

        # User-specific masks in expanded shared space
        self.user_mask = nn.Embedding(U, d_shared)

        # Rx: user-wise attention queries in shared space
        self.user_query = nn.Embedding(U, d_shared)

        # Rx: single-layer linear projection back to BERT space.
        self.rx_proj = nn.Sequential(
            nn.LayerNorm(d_shared),
            nn.Linear(d_shared, d_bert),
        )

        nn.init.normal_(self.user_mask.weight, std=0.5)
        nn.init.normal_(self.user_query.weight, std=0.5)

    def forward(self, bert_embs, snr_db, channel, params=None,
                return_intermediate=False):
        """
        bert_embs: (U, d_bert) - BERT sentence embeddings
        Returns: (U, d_bert) - recovered embeddings
        """
        dev = bert_embs.device

        # ----- Tx: Project to expanded shared space -----
        if params is None:
            e = self.tx_proj(bert_embs)            # (U, d_shared)
            m = self.user_mask.weight              # (U, d_shared)
        else:
            e = functional_call(self.tx_proj, params["tx_proj"],
                                (bert_embs,))
            m = functional_call(
                self.user_mask, params["user_mask"],
                (torch.arange(self.U, device=dev),))

        # Masking in expanded space
        x = e * m                                   # (U, d_shared)
        y_tx = x.sum(dim=0)                         # (d_shared,)

        # ----- Power normalization -----
        # OFDM model: each subcarrier (dimension) has unit power.
        # Larger d_shared = more subcarriers = more bandwidth.
        # Per-dimension SNR is the same regardless of d_shared.
        # This models the bandwidth-quality trade-off:
        #   d_shared=128: compressed (low bandwidth, lossy)
        #   d_shared=768: matched (1:1)
        #   d_shared=3072: expanded (high bandwidth, room for separation)
        y_tx = y_tx / (torch.sqrt((y_tx ** 2).mean() + 1e-8))

        # ----- Channel -----
        y_rx = apply_channel(y_tx, snr_db, channel)

        # ----- Rx: user-wise attention -----
        R = y_rx.unsqueeze(0) * m                   # (U, d_shared)
        Rn = F.normalize(R, p=2, dim=-1)

        recovered = []
        attn_weights = []
        for u in range(self.U):
            if params is None:
                q = self.user_query(torch.tensor(u, device=dev))
            else:
                q = functional_call(
                    self.user_query, params["user_query"],
                    (torch.tensor(u, device=dev),))

            qn = F.normalize(q, p=2, dim=-1)
            # Native cosine-similarity scores (no extra 1/sqrt(d_s)
            # damping). Since Rn, qn are unit-norm, the raw dot product
            # is already in [-1, 1] and softmax-friendly. The standard
            # 1/sqrt(d) attention temperature would collapse softmax to
            # near-uniform when d_s is large (e.g., 3072).
            scores = (Rn @ qn) * 8.0  # mild temperature sharpening
            attn = F.softmax(scores, dim=0)
            attn_weights.append(attn.detach())

            z_u = (attn.unsqueeze(-1) * R).sum(dim=0)  # (d_shared,)

            # Reverse projection: shared space -> BERT space
            if params is None:
                b_hat_u = self.rx_proj(z_u)
            else:
                b_hat_u = functional_call(self.rx_proj, params["rx_proj"],
                                          (z_u,))

            recovered.append(b_hat_u)

        result = torch.stack(recovered, dim=0)  # (U, d_bert)

        if return_intermediate:
            return result, {
                "y_tx": y_tx.detach(),
                "y_rx": y_rx.detach(),
                "attn": torch.stack(attn_weights, dim=0),  # (U, U)
                "masks": m.detach(),
                "projected": e.detach(),
            }

        return result


# =========================================================
# Loss Function
# =========================================================
def semantic_loss(b_orig, b_hat, lam=0.5):
    """
    Combined MSE + Cosine Similarity loss.
    b_orig, b_hat: (U, d_bert)
    """
    mse = F.mse_loss(b_hat, b_orig)
    cos_sim = F.cosine_similarity(b_hat, b_orig, dim=-1).mean()
    loss = (1.0 - lam) * mse + lam * (1.0 - cos_sim)
    return loss, mse.item(), cos_sim.item()


# =========================================================
# MAML Helpers (same structure as original maml.py)
# =========================================================
def split_params(model: BertSemComMux):
    return {
        "tx_proj": dict(model.tx_proj.named_parameters()),
        "user_mask": dict(model.user_mask.named_parameters()),
        "user_query": dict(model.user_query.named_parameters()),
        "rx_proj": dict(model.rx_proj.named_parameters()),
    }


def select_inner_keys(train_mode: str):
    if train_mode == "maml_decoder":
        return ["user_query", "rx_proj"]
    if train_mode == "maml_full":
        return ["tx_proj", "user_mask", "user_query", "rx_proj"]
    return []


def ordered_param_items(param_dict: dict):
    return [(k, param_dict[k]) for k in sorted(param_dict.keys())]


def gather_inner_params(fast_params: dict, inner_keys: list):
    flat_list, meta_index = [], []
    for key in inner_keys:
        for name, p in ordered_param_items(fast_params[key]):
            flat_list.append(p)
            meta_index.append((key, name))
    return flat_list, meta_index


def apply_inner_update(fast_params, inner_keys, meta_index, grads,
                       inner_lr, maml_order):
    new_fast = {k: dict(v) for k, v in fast_params.items()}
    for (key, name), g in zip(meta_index, grads):
        if maml_order == "first":
            g = g.detach()
        new_fast[key][name] = new_fast[key][name] - inner_lr * g
    return new_fast


# =========================================================
# Embedding Cache (pre-compute BERT embeddings)
# =========================================================
class EmbeddingCache:
    """Pre-compute, center, and L2-normalize BERT embeddings.
    Centering removes the dominant mean direction to reduce
    anisotropy; L2 normalization gives unit-norm embeddings
    suitable for cosine-similarity-based loss/metrics.
    """
    def __init__(self, bert_extractor, sentences, batch_size=64):
        self.embeddings = []
        self.sentences = sentences
        print(f"[INFO] Pre-computing BERT embeddings for "
              f"{len(sentences)} sentences...")
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            emb = bert_extractor.encode(batch)
            self.embeddings.append(emb.cpu())
        self.embeddings = torch.cat(self.embeddings, dim=0)

        # Center (subtract mean direction) to reduce anisotropy
        self.mean = self.embeddings.mean(dim=0, keepdim=True)
        self.embeddings = self.embeddings - self.mean
        # L2-normalize
        self.embeddings = F.normalize(self.embeddings, p=2, dim=-1)

        # Sanity check: random pairwise cos sim should be near 0
        n_check = min(500, len(self.embeddings))
        idx1 = torch.randperm(len(self.embeddings))[:n_check]
        idx2 = torch.randperm(len(self.embeddings))[:n_check]
        cs = F.cosine_similarity(self.embeddings[idx1],
                                  self.embeddings[idx2], dim=-1)
        print(f"[INFO] Cached {self.embeddings.shape[0]} embeddings, "
              f"shape={self.embeddings.shape}")
        print(f"[INFO] After centering: random pair cos sim "
              f"mean={cs.mean():.4f}, std={cs.std():.4f}")

    def sample(self, U, device):
        """Sample U random embeddings."""
        idx = torch.randint(0, len(self.embeddings), (U,))
        return self.embeddings[idx].to(device), idx

    def get_sentences(self, idx):
        """Get sentences by index."""
        return [self.sentences[i] for i in idx]


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate(model, cache, args, device):
    model.eval()
    total_cos, total_mse, n_trials = 0.0, 0.0, 0

    if not args.eval_adapt:
        for _ in range(args.eval_trials):
            b, _ = cache.sample(args.users, device)
            b_hat = model(b, args.snr, args.channel, params=None)
            cos = F.cosine_similarity(b_hat, b, dim=-1).mean().item()
            mse = F.mse_loss(b_hat, b).item()
            total_cos += cos
            total_mse += mse
            n_trials += 1
        return total_cos / n_trials, total_mse / n_trials

    # Few-shot adaptation
    inner_keys = select_inner_keys(args.eval_adapt_mode)
    base_params = split_params(model)

    for _ in range(args.eval_trials):
        b_sup, _ = cache.sample(args.users, device)
        fast_params = {k: {n: p for n, p in v.items()}
                       for k, v in base_params.items()}

        with torch.enable_grad():
            for _ in range(args.eval_inner_steps):
                b_hat_sup = model(b_sup, args.snr, args.channel,
                                  params=fast_params)
                loss_sup, _, _ = semantic_loss(b_sup, b_hat_sup,
                                               args.loss_lambda)
                flat_list, meta_index = gather_inner_params(
                    fast_params, inner_keys)
                grads = torch.autograd.grad(loss_sup, flat_list,
                                            create_graph=False)
                fast_params = apply_inner_update(
                    fast_params, inner_keys, meta_index, grads,
                    inner_lr=args.eval_inner_lr, maml_order="first")

        b_q, _ = cache.sample(args.users, device)
        b_hat_q = model(b_q, args.snr, args.channel, params=fast_params)
        cos = F.cosine_similarity(b_hat_q, b_q, dim=-1).mean().item()
        mse = F.mse_loss(b_hat_q, b_q).item()
        total_cos += cos
        total_mse += mse
        n_trials += 1

    return total_cos / n_trials, total_mse / n_trials


# =========================================================
# Train & Eval
# =========================================================
def train_and_eval(args):
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load BERT and pre-compute embeddings ----
    bert = BertEmbeddingExtractor(args.bert_model, device)
    d_bert = bert.embed_dim  # 768

    # Compute d_shared = d_bert * mux_factor
    d_shared = d_bert * args.mux_factor
    print(f"[INFO] d_bert={d_bert}, mux_factor={args.mux_factor}, "
          f"d_shared={d_shared}")

    sentences = load_sentences(args.data_source, args.max_sentences)
    print(f"[INFO] Loaded {len(sentences)} sentences")

    cache = EmbeddingCache(bert, sentences,
                           batch_size=args.bert_batch_size)

    # Free BERT from GPU
    del bert
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Build model ----
    model = BertSemComMux(args.users, d_bert, d_shared,
                          args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_snr_list = list(args.train_snr)
    inner_keys = select_inner_keys(args.train_mode)

    # ---- Config ----
    print("=" * 60)
    print(" BERT Semantic Communication - Expanded Shared Embedding")
    print("=" * 60)
    print(f" Device              : {device}")
    print(f" Train mode          : {args.train_mode}")
    print(f" MAML order          : {args.maml_order}")
    print(f" Users (U)           : {args.users}")
    print(f" BERT dim (d_bert)   : {d_bert}")
    print(f" Mux factor (K)      : {args.mux_factor}")
    print(f" Shared dim (d_shared): {d_shared}")
    print(f" Hidden (MLP)        : {args.hidden}")
    print(f" Channel             : {args.channel}")
    print(f" Loss lambda         : {args.loss_lambda}")
    print("-" * 60)
    print(f" Train SNRs (dB)     : {train_snr_list}")
    print(f" Eval SNR (dB)       : {args.snr}")
    print("-" * 60)
    print(f" Epochs              : {args.epochs}")
    print(f" Steps/epoch         : {args.train_steps_per_epoch}")
    print(f" Eval trials         : {args.eval_trials}")
    if args.train_mode != "joint":
        print("-" * 60)
        print(f" Inner steps         : {args.inner_steps}")
        print(f" Inner LR            : {args.inner_lr}")
        print(f" Meta-batch          : {args.meta_batch}")
        print(f" Inner keys          : {inner_keys}")
    print("=" * 60)

    # ---- CSV ----
    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(
        args.save_dir,
        f"bert_{args.train_mode}_{args.users}U_K{args.mux_factor}.csv"
    )

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_mode", "maml_order",
                "users", "d_bert", "mux_factor", "d_shared",
                "channel", "train_snrs", "eval_snr",
                "inner_steps", "inner_lr", "meta_batch",
                "avg_loss", "cos_sim", "mse"
            ])

    # ---- Training ----
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_meter = 0.0

        for _ in range(args.train_steps_per_epoch):
            if args.train_mode == "joint":
                b, _ = cache.sample(args.users, device)
                snr = float(random.choice(train_snr_list))
                b_hat = model(b, snr, args.channel, params=None)
                loss, _, _ = semantic_loss(b, b_hat, args.loss_lambda)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_meter += loss.item()

            else:
                # MAML
                if args.meta_batch > len(train_snr_list):
                    snr_tasks = [float(random.choice(train_snr_list))
                                 for _ in range(args.meta_batch)]
                else:
                    snr_tasks = [float(x) for x in
                                 random.sample(train_snr_list,
                                               args.meta_batch)]

                base_params = split_params(model)
                meta_loss = torch.tensor(0.0, device=device)

                for snr in snr_tasks:
                    fast_params = {
                        k: {n: p for n, p in v.items()}
                        for k, v in base_params.items()
                    }

                    for _ in range(args.inner_steps):
                        b_sup, _ = cache.sample(args.users, device)
                        b_hat_sup = model(b_sup, snr, args.channel,
                                          params=fast_params)
                        loss_sup, _, _ = semantic_loss(
                            b_sup, b_hat_sup, args.loss_lambda)

                        flat_list, meta_index = gather_inner_params(
                            fast_params, inner_keys)
                        create_graph = (args.maml_order == "second")
                        grads = torch.autograd.grad(
                            loss_sup, flat_list,
                            create_graph=create_graph)
                        fast_params = apply_inner_update(
                            fast_params, inner_keys, meta_index, grads,
                            inner_lr=args.inner_lr,
                            maml_order=args.maml_order)

                    b_q, _ = cache.sample(args.users, device)
                    b_hat_q = model(b_q, snr, args.channel,
                                    params=fast_params)
                    qloss, _, _ = semantic_loss(
                        b_q, b_hat_q, args.loss_lambda)
                    meta_loss = meta_loss + qloss

                meta_loss = meta_loss / float(args.meta_batch)
                opt.zero_grad(set_to_none=True)
                meta_loss.backward()
                opt.step()
                loss_meter += meta_loss.item()

        avg_loss = loss_meter / float(args.train_steps_per_epoch)

        # ---- Eval ----
        cos_sim, mse = evaluate(model, cache, args, device)

        print(f"[Epoch {ep:03d}/{args.epochs}] loss={avg_loss:.4f} | "
              f"SNR={args.snr:.1f}dB | CosSim={cos_sim:.4f} | "
              f"MSE={mse:.4e}")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep, args.train_mode, args.maml_order,
                args.users, d_bert, args.mux_factor, d_shared,
                args.channel, train_snr_list, args.snr,
                (args.inner_steps if args.train_mode != "joint" else ""),
                (args.inner_lr if args.train_mode != "joint" else ""),
                (args.meta_batch if args.train_mode != "joint" else ""),
                avg_loss, cos_sim, mse
            ])

    print(f"\n✅ Results saved to: {csv_path}")
    return model


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BERT-based Multi-User Semantic Communication "
                    "with Expanded Shared Embedding")

    # training mode
    parser.add_argument("--train-mode",
                        choices=["joint", "maml_decoder", "maml_full"],
                        default="joint")
    parser.add_argument("--maml-order", choices=["first", "second"],
                        default="first")

    # model
    parser.add_argument("--users", type=int, default=4)
    parser.add_argument("--mux-factor", type=int, default=4,
                        help="Shared dim = d_bert * mux_factor "
                             "(e.g., 4 -> 768*4=3072)")
    parser.add_argument("--hidden", type=int, default=512)

    # BERT
    parser.add_argument("--bert-model", type=str,
                        default="bert-base-uncased")
    parser.add_argument("--bert-batch-size", type=int, default=64)

    # data
    parser.add_argument("--data-source",
                        choices=["europarl", "ag_news", "synthetic"],
                        default="ag_news")
    parser.add_argument("--max-sentences", type=int, default=50000)

    # channel
    parser.add_argument("--channel", choices=["awgn", "rayleigh"],
                        default="rayleigh")

    # loss
    parser.add_argument("--loss-lambda", type=float, default=0.5)

    # training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-steps-per-epoch", type=int, default=2000)
    parser.add_argument("--train-snr", type=float, nargs="+",
                        default=[0, 5, 10, 15, 20, 25])
    parser.add_argument("--lr", type=float, default=1e-3)

    # MAML
    parser.add_argument("--inner-lr", type=float, default=1e-3)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--meta-batch", type=int, default=4)

    # evaluation
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--eval-trials", type=int, default=5000)

    # test-time adaptation
    parser.add_argument("--eval-adapt", action="store_true")
    parser.add_argument("--eval-adapt-mode",
                        choices=["maml_decoder", "maml_full"],
                        default="maml_decoder")
    parser.add_argument("--eval-inner-steps", type=int, default=1)
    parser.add_argument("--eval-inner-lr", type=float, default=1e-3)

    # misc
    parser.add_argument("--save-dir", type=str, default="results_bert")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug-maml", action="store_true")

    args = parser.parse_args()
    train_and_eval(args)


"""
========================
Example Commands
========================

[1] Joint training (U=4, K=4 -> d_shared=3072, Rayleigh)
python3 bert_semcom.py \
  --train-mode joint \
  --users 4 --mux-factor 4 \
  --snr 10 --channel rayleigh \
  --data-source ag_news \
  --epochs 50 --train-steps-per-epoch 2000 \
  --eval-trials 5000 --cuda

[2] Statistical mux gain: U=2, still K=4 (over-provisioned)
python3 bert_semcom.py \
  --train-mode joint \
  --users 2 --mux-factor 4 \
  --snr 10 --channel rayleigh \
  --epochs 50 --cuda

[3] Compare mux factors: K=1,2,4,8
for K in 1 2 4 8; do
  python3 bert_semcom.py \
    --train-mode joint \
    --users 4 --mux-factor $K \
    --snr 10 --epochs 50 --cuda
done

[4] Decoder-only MAML (U=4, K=4)
python3 bert_semcom.py \
  --train-mode maml_decoder --maml-order first \
  --users 4 --mux-factor 4 \
  --meta-batch 4 --inner-lr 5e-4 --inner-steps 1 \
  --snr 10 --epochs 50 --cuda

[5] Full MAML
python3 bert_semcom.py \
  --train-mode maml_full --maml-order first \
  --users 4 --mux-factor 4 \
  --meta-batch 4 --inner-lr 5e-4 --inner-steps 1 \
  --snr 10 --epochs 50 --cuda

[6] SNR sweep
for snr in 0 5 10 15 20 25 30; do
  python3 bert_semcom.py \
    --train-mode joint \
    --users 4 --mux-factor 4 \
    --snr $snr --epochs 50 --cuda
done

[7] User count comparison (all with K=4)
for u in 2 4 8 16; do
  python3 bert_semcom.py \
    --train-mode joint \
    --users $u --mux-factor 4 \
    --snr 10 --epochs 50 --cuda
done
"""
