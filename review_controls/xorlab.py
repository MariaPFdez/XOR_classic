"""
xorlab.py — controlled re-implementation of the training rules in XORclassic/
(this repository, github.com/MariaPFdez/XOR_classic) with the switches needed
for the control experiments.

Default settings are bit-identical to XORclassic/ (checked by
test_equivalence.py; the CNN per-sample loops are vectorised with einsum, but
the summed terms are integer-valued, so the sums are exact):
  * binarisation x > 0.5, Theta(u) = 1 if u >= 0
  * W, V ~ N(0, 0.01^2), biases 0, B ~ N(0, 1/D)   (FC); same RNG call order
    (W, V, B), so a given seed gives the same initialisation as the repo
  * eta_dec=0.05, eta_enc=0.02, eta_b_dec=0.02, eta_b_enc=0.01
  * weight updates summed over the batch; bias updates averaged over the
    batch (FC) or over batch and spatial positions (CNN)
  * up to max_corrections=3 weight updates on the same mini-batch
  * CNN: 28 3x3 filters, 4x4 max-pool, max-unpool with encoder indices,
    single 3x3 decoder filter; the encoder feedback sign(conv(e, B)) is
    applied at every pre-pooling position (not gated by the pooling winners)
  * readout: multi-class perceptron, eta 0.05 / 0.02, 5 epochs, argmax
Differences from the repo are only additions (seeds, logging, options).
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, asdict, field

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------


@dataclass
class Config:
    name: str = "fc_ref"
    arch: str = "fc"                 # fc | cnn
    rule: str = "xor"                # xor | ste  (ste = backprop-through-STE baseline)
    error: str = "signed"            # signed | unsigned            (xor rule only)
    enc_feedback: str = "sign"       # sign | raw | raw_norm        (xor rule only)
                                     # raw_norm: e·B rescaled so mean|.| = 1, the step size of sign
    freeze_encoder: bool = False     # decoder-only learning (random fixed encoder)
    freeze_W: bool = False           # learn encoder biases only (W fixed)
    enc_bias_init: float = 0.0       # initial encoder bias (0 = repo)
    bias_mode: str = "mean"          # mean (repo) | sum | none
    hidden: int = 1024               # FC hidden units
    channels: int = 28               # CNN channels
    upsample: str = "indices"        # indices (repo) | nearest      (CNN)
    train_data: str = "MNIST"        # MNIST | Random
    epochs: int = 5
    max_corrections: int = 3
    batch_size: int = 256
    eta_dec: float = 0.05
    eta_enc: float = 0.02
    eta_b_dec: float = 0.02
    eta_b_enc: float = 0.01
    ste_lr: float = 1e-3
    readout_epochs: int = 5
    eta_cls: float = 0.05
    eta_b_cls: float = 0.02
    lpips: bool = False
    save_weights: bool = False
    seed: int = 0
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# runtime setup
# ----------------------------------------------------------------------------


def setup_torch(seed: int):
    torch.manual_seed(seed)
    # TF32 can flip threshold decisions; keep full fp32 for fidelity
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# data (whole datasets held on device as uint8; batches converted on the fly)
# ----------------------------------------------------------------------------

_DATA_CACHE: dict = {}


def _fake(n, seed, k, proto_seed):
    """Structured synthetic data for smoke tests (XORLAB_FAKE=1): k fixed
    prototypes per dataset (shared by train/test), 3% pixel noise."""
    protos = torch.rand(k, 28, 28, generator=torch.Generator().manual_seed(proto_seed)) < 0.13
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, k, (n,), generator=g)
    noise = torch.rand(n, 28, 28, generator=g) < 0.03
    X = (protos[y] ^ noise).to(torch.uint8) * 255
    return X, y


def load_split(name: str, train: bool, root: str, emnist_transpose=False):
    key = (name, train, emnist_transpose)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    if os.environ.get("XORLAB_FAKE") == "1":
        k = 10 if name == "MNIST" else 47
        n = (2000 if train else 500)
        ps = {'MNIST': 11, 'EMNIST': 23}[name]
        X, y = _fake(n, ps + 100 * int(train), k, ps)
    else:
        from torchvision import datasets
        if name == "MNIST":
            ds = datasets.MNIST(root, train=train, download=True)
        elif name == "EMNIST":
            ds = datasets.EMNIST(root, split="balanced", train=train, download=True)
        else:
            raise ValueError(name)
        X, y = ds.data, ds.targets
        if name == "EMNIST" and emnist_transpose:
            X = X.transpose(1, 2).contiguous()
    Xb = (X.float() / 255.0 > 0.5).to(torch.uint8)      # identical to repo tfm
    out = (Xb, y.long())
    _DATA_CACHE[key] = out
    return out


def mnist_p_white(root):
    Xtr, _ = load_split("MNIST", True, root)
    Xte, _ = load_split("MNIST", False, root)
    tot = Xtr.numel() + Xte.numel()
    return (Xtr.sum().item() + Xte.sum().item()) / tot


def random_split(train: bool, root: str, seed: int):
    """Matched-density Bernoulli images, as in repo create_random_dataset."""
    p = mnist_p_white(root)
    n = 60000 if train else 10000
    if os.environ.get("XORLAB_FAKE") == "1":
        n = 2000 if train else 500
    g = torch.Generator().manual_seed(seed)
    X = (torch.rand(n, 28, 28, generator=g) < p).to(torch.uint8)
    return X, torch.zeros(n, dtype=torch.long)


def get_data(name, train, root, seed=0):
    if name == "Random":
        # train set depends on run seed; test set is fixed across runs
        return random_split(train, root, seed=(1000 + seed) if train else 424242)
    return load_split(name, train, root)


def batches(n, bs, gen, shuffle=True):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------


class FCModel:
    def __init__(self, cfg: Config, dev):
        D, H = 784, cfg.hidden
        self.cfg = cfg
        self.W = torch.randn(D, H, device=dev) * 0.010
        self.bW = torch.full((H,), cfg.enc_bias_init, device=dev)
        self.V = torch.randn(H, D, device=dev) * 0.010
        self.bV = torch.zeros(D, device=dev)
        self.B = torch.randn(D, H, device=dev) * (1.0 / math.sqrt(D))
        self.W0, self.bW0 = self.W.clone(), self.bW.clone()

    def prep(self, xb):                       # uint8 (N,28,28) -> float (N,784)
        return xb.view(xb.shape[0], -1).float()

    def encode(self, x, initial=False):
        W, b = (self.W0, self.bW0) if initial else (self.W, self.bW)
        return ((x @ W + b) >= 0.0).float()

    def decode(self, h):
        return ((h @ self.V + self.bV) >= 0.0).float()

    def forward(self, x):
        h = self.encode(x)
        return h, self.decode(h)

    def features(self, x, initial=False):
        return self.encode(x, initial)

    def n_params(self):
        return sum(t.numel() for t in (self.W, self.bW, self.V, self.bV))

    def update(self, x, h, y, delta, enc_err):
        c = self.cfg
        self.V += c.eta_dec * (h.t() @ delta)
        _bias(self, "bV", c.eta_b_dec, delta, (0,), c.bias_mode)
        if not c.freeze_encoder:
            enc = _feedback(enc_err @ self.B, c.enc_feedback)
            if not c.freeze_W:
                self.W += c.eta_enc * (x.t() @ enc)
            _bias(self, "bW", c.eta_b_enc, enc, (0,), c.bias_mode)

    def state(self):
        return {k: getattr(self, k).cpu() for k in ("W", "bW", "V", "bV", "B")}


class CNNModel:
    def __init__(self, cfg: Config, dev):
        C, k = cfg.channels, 3
        self.cfg, self.k, self.pad = cfg, k, 1
        self.W = torch.randn(C, 1, k, k, device=dev) * 0.010
        self.bW = torch.full((1, C, 1, 1), cfg.enc_bias_init, device=dev)
        self.V = torch.randn(1, C, k, k, device=dev) * 0.010
        self.bV = torch.zeros(1, 1, 1, 1, device=dev)
        self.B = torch.randn(C, 1, k, k, device=dev) * (1.0 / math.sqrt(k * k))
        self.W0, self.bW0 = self.W.clone(), self.bW.clone()

    def prep(self, xb):
        return xb.view(-1, 1, 28, 28).float()

    def _enc(self, x, initial=False):
        W, b = (self.W0, self.bW0) if initial else (self.W, self.bW)
        hid = ((F.conv2d(x, W, padding=self.pad) + b) >= 0.0).float()
        pooled, idx = F.max_pool2d(hid, 4, return_indices=True)
        return pooled, idx

    def _up(self, pooled, idx):
        if self.cfg.upsample == "indices":
            return F.max_unpool2d(pooled, idx, 4)
        return F.interpolate(pooled, scale_factor=4, mode="nearest")

    def forward(self, x):
        pooled, idx = self._enc(x)
        hid = self._up(pooled, idx)
        y = ((F.conv2d(hid, self.V, padding=self.pad) + self.bV) >= 0.0).float()
        return hid, y

    def features(self, x, initial=False):
        pooled, _ = self._enc(x, initial)
        return pooled.flatten(1)

    def n_params(self):
        return sum(t.numel() for t in (self.W, self.bW, self.V, self.bV))

    def update(self, x, hid, y, delta, enc_err):
        c, k, p = self.cfg, self.k, self.pad
        N = x.shape[0]
        # decoder: identical to repo decoder_update, vectorised
        hid_unf = F.unfold(hid, kernel_size=k, padding=p)              # N, C*k*k, L
        d_unf = delta.view(N, 1, -1)                                   # N, 1, L
        dV = torch.einsum("bol,bkl->ok", d_unf, hid_unf).view_as(self.V)
        self.V += c.eta_dec * dV
        _bias(self, "bV", c.eta_b_dec, delta, (0, 2, 3), c.bias_mode, keepdim=True)
        if not c.freeze_encoder:
            # feedback at every pre-pooling position, as in the repo
            enc = _feedback(F.conv2d(enc_err, self.B, padding=p), c.enc_feedback)   # N, C, 28, 28
            inp_unf = F.unfold(x, kernel_size=k, padding=p)            # N, k*k, L
            dW = torch.einsum("bcl,bkl->ck", enc.flatten(2), inp_unf).view_as(self.W)
            if not c.freeze_W:
                self.W += c.eta_enc * dW
            _bias(self, "bW", c.eta_b_enc, enc, (0, 2, 3), c.bias_mode, keepdim=True)

    def state(self):
        return {k: getattr(self, k).cpu() for k in ("W", "bW", "V", "bV", "B")}


def _feedback(proj, mode):
    """Encoder teaching signal from the random projection e·B.
    sign (repo): ±1 per unit.  raw: graded e·B (much smaller steps, since
    |e·B| << 1 with B ~ N(0, 1/D)).  raw_norm: graded, rescaled per update so
    that mean|.| = 1, i.e. the same average step size as sign."""
    if mode == "sign":
        return torch.sign(proj)
    if mode == "raw":
        return proj
    if mode == "raw_norm":
        return proj / proj.abs().mean().clamp_min(1e-12)
    raise ValueError(mode)


def _bias(m, name, eta, err, dims, mode, keepdim=False):
    if mode == "none":
        return
    red = err.mean(dim=dims, keepdim=keepdim) if mode == "mean" else err.sum(dim=dims, keepdim=keepdim)
    getattr(m, name).add_(eta * red)


def xor_error(x, y, mode):
    """signed: XOR with sign exactly as in the repo (m_up - m_dn = x - y).
    unsigned: |x - y|, a magnitude-only control. It carries no direction, so
    the update can only push outputs up and must collapse to all-ones; it
    shows that the sign is logically required."""
    e = (x != y)
    xb = x.bool()
    if mode == "signed":
        return (e & xb).float() - (e & ~xb).float()
    return e.float()


# ----------------------------------------------------------------------------
# straight-through-estimator baseline (backprop upper bound, same architecture)
# ----------------------------------------------------------------------------


class _STEHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return (a >= 0).float()

    @staticmethod
    def backward(ctx, g):
        (a,) = ctx.saved_tensors
        return g * (a.abs() <= 1).float()


def train_ste(model, cfg, Xtr, dev, gen, log):
    params = [model.W, model.bW, model.V, model.bV]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=cfg.ste_lr)
    n_upd = 0
    for ep in range(cfg.epochs):
        loss_sum, nb = 0.0, 0
        for bi in batches(Xtr.shape[0], cfg.batch_size, gen):
            x = model.prep(Xtr[bi].to(dev))
            if cfg.arch == "fc":
                h = _STEHeaviside.apply(x @ model.W + model.bW)
                z = h @ model.V + model.bV
            else:
                hid = _STEHeaviside.apply(F.conv2d(x, model.W, padding=1) + model.bW)
                pooled, idx = F.max_pool2d(hid, 4, return_indices=True)
                up = model._up(pooled, idx)
                z = F.conv2d(up, model.V, padding=1) + model.bV
            loss = F.binary_cross_entropy_with_logits(z, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            n_upd += 1
            loss_sum += loss.item(); nb += 1
        # loss: last batch (kept for compatibility); loss_epoch_mean: used to
        # pick the STE learning rate on training data only (paper_numbers.py)
        log.append({"epoch": ep + 1, "loss": loss.item(), "loss_epoch_mean": loss_sum / nb})
    for p in params:
        p.requires_grad_(False)
    return n_upd


# ----------------------------------------------------------------------------
# XOR training loop (mirror of training_fully / training_cnn)
# ----------------------------------------------------------------------------


@torch.no_grad()
def train_xor(model, cfg, Xtr, Xte, dev, gen, log):
    n_upd = 0
    for ep in range(cfg.epochs):
        t0 = time.time()
        pre_ok = post_ok = tot = 0
        corr_hist = [0] * (cfg.max_corrections + 1)
        for bi in batches(Xtr.shape[0], cfg.batch_size, gen):
            x = model.prep(Xtr[bi].to(dev))
            h, y = model.forward(x)
            pre_ok += (x == y).sum().item()               # online (pre-update) accuracy
            count = 0
            while (x != y).any() and count < cfg.max_corrections:
                delta = xor_error(x, y, cfg.error)
                model.update(x, h, y, delta, delta)
                h, y = model.forward(x)
                count += 1
                n_upd += 1
            corr_hist[count] += 1
            post_ok += (x == y).sum().item()               # repo's printed "Accuracy"
            tot += x.numel()
        te = recon_metrics(model, Xte, dev, cfg.batch_size)
        log.append({"epoch": ep + 1,
                    "train_acc_pre_update": pre_ok / tot,
                    "train_acc_post_corrections": post_ok / tot,
                    "test_bit_acc": te["bit_acc"],
                    "test_perfect_frac": te["perfect_frac"],
                    "corrections_hist": corr_hist,
                    "updates_so_far": n_upd,
                    "sec": time.time() - t0})
    return n_upd


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------


@torch.no_grad()
def recon_metrics(model, X, dev, bs=1024, lpips_fn=None):  # 1024 = 4x256 keeps repo batch boundaries
    TP = TN = FP = FN = 0
    perfect = 0
    ham = 0.0
    n = X.shape[0]
    lp_fixed, lp_repo = [], []
    for i in range(0, n, bs):
        x = model.prep(X[i:i + bs].to(dev))
        _, y = model.forward(x)
        xb, yb = x.bool(), y.bool()
        TP += (xb & yb).sum().item(); TN += (~xb & ~yb).sum().item()
        FP += (~xb & yb).sum().item(); FN += (xb & ~yb).sum().item()
        err = (xb != yb).view(x.shape[0], -1).sum(1)
        perfect += (err == 0).sum().item()
        ham += err.float().sum().item()
        if lpips_fn is not None:
            a = x.view(-1, 1, 28, 28).repeat(1, 3, 1, 1)
            b = y.view(-1, 1, 28, 28).repeat(1, 3, 1, 1)
            d = lpips_fn(a, b, normalize=True).flatten()           # all images, [0,1]->[-1,1]
            lp_fixed.append(d.cpu())
            # repo behaviour: inputs not normalised, only first image of each 256-batch
            for j in range(0, a.shape[0], 256):
                lp_repo.append(lpips_fn(a[j:j + 1], b[j:j + 1]).flatten().cpu())
    tot = TP + TN + FP + FN
    prec = TP / max(1, TP + FP); rec = TP / max(1, TP + FN)
    out = {
        "bit_acc": (TP + TN) / tot,
        "balanced_acc": 0.5 * (TP / max(1, TP + FN) + TN / max(1, TN + FP)),
        "f1_pos": 2 * prec * rec / max(1e-9, prec + rec),
        "precision_pos": prec, "recall_pos": rec,
        "ink_frac": (TP + FN) / tot,
        "allzero_bit_acc": (TN + FP) / tot,
        "perfect_frac": perfect / n,
        "hamming_per_image": ham / n,
    }
    if lpips_fn is not None:
        out["lpips_fixed"] = torch.cat(lp_fixed).mean().item()
        out["lpips_repo_style"] = torch.cat(lp_repo).mean().item()
    return out


@torch.no_grad()
def feats(fn, X, prep, dev, bs=2000):
    return torch.cat([fn(prep(X[i:i + bs].to(dev))) for i in range(0, X.shape[0], bs)])


@torch.no_grad()
def perceptron_readout(Ftr, ytr, Fte, yte, K, cfg, gen, dev, return_weights=False):
    """Mirror of training_*_classif: multi-class perceptron on argmax."""
    C = torch.zeros(Ftr.shape[1], K, device=dev)
    bC = torch.zeros(K, device=dev)
    ytr, yte = ytr.to(dev), yte.to(dev)
    for _ in range(cfg.readout_epochs):
        for bi in batches(Ftr.shape[0], cfg.batch_size, gen):
            f, t = Ftr[bi.to(dev)], ytr[bi.to(dev)]
            pred = (f @ C + bC).argmax(1)
            err = (F.one_hot(t, K) - F.one_hot(pred, K)).float()
            C += cfg.eta_cls * (f.t() @ err)
            bC += cfg.eta_b_cls * err.mean(0)
    pred = (Fte @ C + bC).argmax(1)
    acc = (pred == yte).float().mean().item()
    per_class = [((pred == yte) & (yte == k)).sum().item() / max(1, (yte == k).sum().item())
                 for k in range(K)]
    if return_weights:                      # used by test_equivalence.py
        return acc, per_class, C, bC
    return acc, per_class


@torch.no_grad()
def ridge_readout(Ftr, ytr, Fte, yte, K, dev, gen):
    """Best-linear reference: ridge regression to one-hot, lambda chosen on a
    10% split of the training set (never on test)."""
    n = Ftr.shape[0]
    perm = torch.randperm(n, generator=gen).to(dev)
    nv = n // 10
    va, tr = perm[:nv], perm[nv:]

    def acc(Wr, Fb, yb):
        A = torch.cat([Fb, torch.ones(Fb.shape[0], 1, device=dev)], 1).double()
        return ((A @ Wr).argmax(1) == yb).float().mean().item()

    ytr, yte = ytr.to(dev), yte.to(dev)
    lams = (1e-2, 1e-1, 1, 10, 100, 1000)

    def gram(Fa, ya):
        A = torch.cat([Fa, torch.ones(Fa.shape[0], 1, device=dev)], 1).double()
        return A.t() @ A, A.t() @ F.one_hot(ya, K).double()

    def solve(G, AY, lam):
        return torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=dev, dtype=G.dtype), AY)

    Gtr, AYtr = gram(Ftr[tr], ytr[tr])
    best = max((acc(solve(Gtr, AYtr, lam), Ftr[va], ytr[va]), lam) for lam in lams)
    Gall, AYall = gram(Ftr, ytr)
    Wr = solve(Gall, AYall, best[1])
    return acc(Wr, Fte, yte), best[1]


# ----------------------------------------------------------------------------
# one full run
# ----------------------------------------------------------------------------


def run(cfg: Config, data_root: str, do_readout=True, raw_baseline=False):
    dev = setup_torch(cfg.seed)
    t_start = time.time()
    model = FCModel(cfg, dev) if cfg.arch == "fc" else CNNModel(cfg, dev)
    Xtr, _ = get_data(cfg.train_data, True, data_root, cfg.seed)
    Xte_same, _ = get_data(cfg.train_data, False, data_root, cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    eval_sets = {"MNIST": get_data("MNIST", False, data_root)[0],
                 "EMNIST": get_data("EMNIST", False, data_root)[0],
                 "Random": get_data("Random", False, data_root)[0]}

    lpips_fn = None
    if cfg.lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()

    before = {k: recon_metrics(model, X, dev) for k, X in eval_sets.items()}

    log = []
    if cfg.rule == "xor":
        n_upd = train_xor(model, cfg, Xtr, Xte_same, dev, gen, log)
    else:
        n_upd = train_ste(model, cfg, Xtr, dev, gen, log)

    after = {k: recon_metrics(model, X, dev, lpips_fn=lpips_fn) for k, X in eval_sets.items()}

    res = {"config": asdict(cfg), "device": str(dev),
           "torch": torch.__version__, "n_params": model.n_params(),
           "weight_updates": n_upd, "log": log,
           "recon_before": before, "recon_after": after}

    if do_readout:
        ro = {}
        for ds, K in (("MNIST", 10), ("EMNIST", 47)):
            Xa, ya = get_data(ds, True, data_root)
            Xb, yb = get_data(ds, False, data_root)
            for tag, initial in (("trained", False), ("init_encoder", True)):
                Ftr = feats(lambda x: model.features(x, initial), Xa, model.prep, dev)
                Fte = feats(lambda x: model.features(x, initial), Xb, model.prep, dev)
                g2 = torch.Generator().manual_seed(cfg.seed + 7)
                acc, pc = perceptron_readout(Ftr, ya, Fte, yb, K, cfg, g2, dev)
                racc, lam = ridge_readout(Ftr, ya, Fte, yb, K, dev, g2)
                ro[f"{ds}_{tag}"] = {"perceptron_acc": acc, "per_class": pc,
                                     "ridge_acc": racc, "ridge_lambda": lam,
                                     "n_features": Ftr.shape[1],
                                     "active_frac": Ftr.float().mean().item()}
                del Ftr, Fte
            if raw_baseline:
                prep = lambda xb: xb.view(xb.shape[0], -1).float()
                Ftr = feats(lambda x: x, Xa, prep, dev)
                Fte = feats(lambda x: x, Xb, prep, dev)
                g2 = torch.Generator().manual_seed(cfg.seed + 7)
                acc, pc = perceptron_readout(Ftr, ya, Fte, yb, K, cfg, g2, dev)
                racc, lam = ridge_readout(Ftr, ya, Fte, yb, K, dev, g2)
                ro[f"{ds}_raw_pixels"] = {"perceptron_acc": acc, "per_class": pc,
                                          "ridge_acc": racc, "ridge_lambda": lam,
                                          "n_features": 784}
        res["readout"] = ro

    res["wall_sec"] = time.time() - t_start
    return res, (model.state() if cfg.save_weights else None)
