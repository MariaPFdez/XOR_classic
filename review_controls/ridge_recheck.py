#!/usr/bin/env python3
"""
ridge_recheck.py — is the ridge (least-squares) readout limited by the λ grid?

In the grid runs λ is chosen on a 10% validation split from {1e-2, ..., 1e3}.
For raw pixels the choice is 1e3 (the upper edge) in every seed.  This script
recomputes the readout on an extended grid (1e-2 ... 1e6, half-decades) for the
encoders that can be rebuilt without re-training:

  raw pixels                      (from A_fc_ref)
  FC  trained encoder             A_fc_ref   (results/A_fc_ref/seed<k>.weights.pt)
  FC  initial / fixed encoder     = C_fc_frozen_enc, rebuilt from the seed
  CNN trained encoder             A_cnn_ref  (results/A_cnn_ref/seed<k>.weights.pt)
  CNN initial / fixed encoder     = C_cnn_frozen_enc, rebuilt from the seed

The validation split is the one used in the run (same generator, same number of
draws), so the value stored in the JSON is reproduced first as a check.

    python ridge_recheck.py [--out results] [--seeds 0-4]  > ridge_recheck.txt
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xorlab  # noqa: E402
from xorlab import Config  # noqa: E402

OLD = (1e-2, 1e-1, 1, 10, 100, 1000)
NEW = tuple(10 ** (k / 2) for k in range(-4, 13))          # 1e-2 ... 1e6, half-decades


@torch.no_grad()
def path(Ftr, ytr, Fte, yte, K, dev, perm, lams):
    """val and test accuracy for every λ (eigendecomposition: one solve per λ is cheap)."""
    n = Ftr.shape[0]
    nv = n // 10
    va, tr = perm[:nv], perm[nv:]
    aug = lambda Fb: torch.cat([Fb, torch.ones(Fb.shape[0], 1, device=dev)], 1).double()

    def fit_all(A, Y):
        G, AY = A.t() @ A, A.t() @ Y
        ev, Q = torch.linalg.eigh(G)
        QAY = Q.t() @ AY
        return lambda lam: Q @ (QAY / (ev + lam).unsqueeze(1))

    acc = lambda Wr, Fb, yb: ((aug(Fb) @ Wr).argmax(1) == yb).float().mean().item()
    f_tr = fit_all(aug(Ftr[tr]), F.one_hot(ytr[tr], K).double())
    f_all = fit_all(aug(Ftr), F.one_hot(ytr, K).double())
    return {lam: (acc(f_tr(lam), Ftr[va], ytr[va]), acc(f_all(lam), Fte, yte)) for lam in lams}


def pick(curve, lams):
    best = max((curve[l][0], l) for l in lams)      # same tie rule as xorlab.ridge_readout
    return best[1], curve[best[1]][1]


def split_like_run(n, seed, readout_epochs, dev):
    """Reproduce g2 in xorlab.run: perceptron_readout draws one randperm per epoch,
    then ridge_readout draws the validation permutation."""
    g2 = torch.Generator().manual_seed(seed + 7)
    for _ in range(readout_epochs):
        torch.randperm(n, generator=g2)
    return torch.randperm(n, generator=g2).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--data-root", default=os.environ.get("XORLAB_DATA", "./data"))
    ap.add_argument("--seeds", default="0-4")
    args = ap.parse_args()
    a, b = (int(x) for x in args.seeds.split("-")) if "-" in args.seeds else (int(args.seeds),) * 2
    seeds = range(a, b + 1)
    dev = xorlab.setup_torch(0)
    data = {ds: (xorlab.load_split(ds, True, args.data_root), xorlab.load_split(ds, False, args.data_root))
            for ds in ("MNIST", "EMNIST")}
    lams = sorted(set(OLD) | set(NEW))

    print(f"{'encoder':22s} {'data':6s} seed  stored(λ)        recomputed  | extended: λ*     test   "
          f"Δ vs stored | val range over λ")
    summary = {}
    for seed in seeds:
        jf = os.path.join(args.out, "A_fc_ref", f"seed{seed}.json")
        jc = os.path.join(args.out, "A_cnn_ref", f"seed{seed}.json")
        with open(jf) as f:
            Rf = json.load(f)["readout"]
        with open(jc) as f:
            Rc = json.load(f)["readout"]

        encoders = []
        # FC
        xorlab.setup_torch(seed)
        mf = xorlab.FCModel(Config(arch="fc", seed=seed), dev)
        sf = torch.load(jf.replace(".json", ".weights.pt"), map_location=dev)
        for k, v in sf.items():
            setattr(mf, k, v.to(dev))
        encoders += [("raw pixels", lambda x: x, lambda xb: xb.view(xb.shape[0], -1).float(), Rf, "raw_pixels"),
                     ("FC trained", lambda x, m=mf: m.features(x), mf.prep, Rf, "trained"),
                     ("FC fixed (initial)", lambda x, m=mf: m.features(x, True), mf.prep, Rf, "init_encoder")]
        # CNN
        xorlab.setup_torch(seed)
        mc = xorlab.CNNModel(Config(arch="cnn", seed=seed), dev)
        sc = torch.load(jc.replace(".json", ".weights.pt"), map_location=dev)
        for k, v in sc.items():
            setattr(mc, k, v.to(dev))
        encoders += [("CNN trained", lambda x, m=mc: m.features(x), mc.prep, Rc, "trained"),
                     ("CNN fixed (initial)", lambda x, m=mc: m.features(x, True), mc.prep, Rc, "init_encoder")]

        for name, fn, prep, R, tag in encoders:
            for ds, K in (("MNIST", 10), ("EMNIST", 47)):
                (Xa, ya), (Xb, yb) = data[ds]
                Ftr = xorlab.feats(fn, Xa, prep, dev)
                Fte = xorlab.feats(fn, Xb, prep, dev)
                perm = split_like_run(Ftr.shape[0], seed, 5, dev)
                curve = path(Ftr, ya.to(dev), Fte, yb.to(dev), K, dev, perm, lams)
                lo, to = pick(curve, OLD)
                ln, tn = pick(curve, lams)
                st = R[f"{ds}_{tag}"]
                vals = [curve[l][0] for l in lams]
                print(f"{name:22s} {ds:6s} {seed:4d}  {100 * st['ridge_acc']:6.2f} ({st['ridge_lambda']:g})"
                      f"{'':>4s}{100 * to:6.2f} ({lo:g}) | {ln:9.3g}  {100 * tn:6.2f}  {100 * (tn - st['ridge_acc']):+6.2f}"
                      f"      | {100 * min(vals):.2f}–{100 * max(vals):.2f}", flush=True)
                summary.setdefault((name, ds), []).append((st["ridge_acc"], tn, ln))
                del Ftr, Fte

    print("\nMean over seeds (test %): stored grid → extended grid")
    for (name, ds), v in summary.items():
        s = [x[0] for x in v]; t = [x[1] for x in v]
        m = lambda z: 100 * sum(z) / len(z)
        sd = lambda z: 100 * math.sqrt(sum((x - sum(z) / len(z)) ** 2 for x in z) / max(1, len(z) - 1))
        print(f"  {name:22s} {ds:6s} {m(s):6.2f} ± {sd(s):.2f}  →  {m(t):6.2f} ± {sd(t):.2f}   "
              f"λ* {' '.join(f'{x[2]:.3g}' for x in v)}")


if __name__ == "__main__":
    main()
