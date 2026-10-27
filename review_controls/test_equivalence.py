#!/usr/bin/env python3
"""
test_equivalence.py — check that xorlab.py applies the same rules as XORclassic/.

    python test_equivalence.py [--repo ..] [--n 2048] [--epochs 2] [--seed 0]
    XORLAB_FAKE=1 python test_equivalence.py          # synthetic data, no download

Both implementations see the same images in the same fixed order (no
shuffling), start from the same seed and run with the same backend flags
(TF32 off, deterministic cuDNN).  XORclassic/ must be importable (default:
the parent directory of review_controls/); it needs torchvision.

Checks                                                       criterion
  data     binarisation and EMNIST orientation (real data)   identical
  FC       W, b_W, V, b_V, B after --epochs of training      bit-identical
  FC       perceptron readout C, b_C on the same encoder     bit-identical
  CNN      forward pass (hidden, unpooled, reconstruction)   bit-identical
  CNN      one update (V, b_V, W, b_W) from the same state   rel. diff < 1e-5
           (at initialisation and after training)
  CNN      perceptron readout C, b_C on the same encoder     bit-identical
  CNN      --epochs of training                              reported only (WARN,
           not FAIL): einsum and the per-sample loop sum in a different order.
           The summed terms are integer-valued (±1 x {0,1}), so the sums are
           normally exact and the runs identical, but a single fp32 difference
           at a threshold u ~ 0 would let the two runs drift apart.
Exit code 0 = PASS.
"""
import argparse
import contextlib
import io
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import xorlab  # noqa: E402
from xorlab import Config  # noqa: E402

RESULTS = []


def report(name, ok, detail="", soft=False):
    tag = "ok" if ok else ("WARN" if soft else "FAIL")
    RESULTS.append((tag, name))
    print(f"[{tag:4s}] {name}" + (f"  — {detail}" if detail else ""), flush=True)


def exact(name, pairs):
    bad = [(k, (a - b).abs().max().item()) for k, a, b in pairs if not torch.equal(a, b)]
    report(name, not bad, "identical" if not bad else
           "max |diff|: " + ", ".join(f"{k} {d:.3g}" for k, d in bad))


def rel_close(name, pairs, ref, tol=1e-5):
    """max |a - b| relative to the size of the update (a, b: weights after the
    update; ref: weights before it)."""
    worst = []
    for k, a, b, r in [(k, a, b, ref[k]) for k, a, b in pairs]:
        scale = max((a - r).abs().max().item(), 1e-30)
        worst.append((k, (a - b).abs().max().item() / scale))
    ok = all(w < tol for _, w in worst)
    if all(w == 0 for _, w in worst):
        report(name, ok, "bit-identical")
    else:
        report(name, ok, "rel. diff " + ", ".join(f"{k} {w:.2g}" for k, w in worst))


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def loader(X, y, bs=256):
    ds = torch.utils.data.TensorDataset(X.float().unsqueeze(1), y)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.dirname(HERE),
                    help="directory that contains XORclassic/ (default: parent of review_controls/)")
    ap.add_argument("--data-root", default=os.environ.get("XORLAB_DATA", "./data"))
    ap.add_argument("--n", type=int, default=2048, help="training images used")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = os.path.abspath(args.data_root)

    sys.path.insert(0, os.path.abspath(args.repo))
    with contextlib.redirect_stdout(io.StringIO()):
        import XORclassic.fully_conn as rfc
        import XORclassic.cnn as rcnn
        import XORclassic.utils as rut
    dev = rfc.device
    print(f"XORclassic from {os.path.abspath(args.repo)}  device {dev}  torch {torch.__version__}"
          f"{'  (XORLAB_FAKE=1: synthetic data)' if os.environ.get('XORLAB_FAKE') == '1' else ''}\n")

    # fixed batch order on the xorlab side (the repo side uses shuffle=False loaders)
    _orig_batches = xorlab.batches
    xorlab.batches = lambda n, bs, gen, shuffle=True: _orig_batches(n, bs, gen, shuffle=False)

    Xall, yall = xorlab.load_split("MNIST", True, root)
    n = min(args.n, Xall.shape[0])
    X, y = Xall[:n], yall[:n]
    Xte, yte = xorlab.load_split("MNIST", False, root)
    Xte, yte = Xte[:1024], yte[:1024]
    E = args.epochs
    gen = lambda: torch.Generator().manual_seed(args.seed)

    # ---------------- data ----------------
    if os.environ.get("XORLAB_FAKE") != "1":
        from torchvision import datasets
        for name, kw in (("MNIST", {}), ("EMNIST", {"split": "balanced"})):
            ds = getattr(datasets, name)(root, train=True, download=False, transform=rut.tfm, **kw)
            rep = torch.stack([ds[i][0] for i in range(512)]).squeeze(1)
            mine = xorlab.load_split(name, True, root)[0][:512].float()
            exact(f"data: {name} binarisation/orientation (512 images)", [("x", rep, mine)])

    # ---------------- FC training ----------------
    rfc.epochs = E
    xorlab.setup_torch(args.seed)
    ref = quiet(rfc.training_fully, loader(X, y), loader(Xte, yte))
    xorlab.setup_torch(args.seed)
    cfg = Config(arch="fc", epochs=E, readout_epochs=E)
    m = xorlab.FCModel(cfg, dev)
    xorlab.train_xor(m, cfg, X, Xte, dev, gen(), [])
    exact(f"FC training, {E} epochs x {n} images",
          [(k, ref[r], getattr(m, k)) for k, r in (("W", "W"), ("bW", "b_W"), ("V", "V"), ("bV", "b_V"), ("B", "B"))])

    # ---------------- readouts (same encoder on both sides) ----------------
    def readout_check(label, network, model, ref_vars, rmod):
        for k, r in (("W", "W"), ("bW", "b_W"), ("V", "V"), ("bV", "b_V"), ("B", "B")):
            setattr(model, k, ref_vars[r].clone())
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                quiet(rut.save_variables, network, "MNIST", ref_vars)
                rc = quiet(getattr(rmod, "training_fully_classif" if network == "Fully" else "training_cnn_classif"),
                           loader(X, y), loader(Xte, yte), "MNIST", "MNIST", network)
            finally:
                os.chdir(cwd)
        Ftr = xorlab.feats(lambda x: model.features(x), X, model.prep, dev)
        Fte = xorlab.feats(lambda x: model.features(x), Xte, model.prep, dev)
        _, _, C, bC = xorlab.perceptron_readout(Ftr, y, Fte, yte, 10, model.cfg, gen(), dev, return_weights=True)
        exact(f"{label} perceptron readout, {E} epochs", [("C", rc["C"], C), ("b_C", rc["b_C"], bC)])

    readout_check("FC", "Fully", m, ref, rfc)

    # ---------------- CNN ----------------
    rcnn.epochs = 0                                   # 0 epochs: the repo returns its initial weights
    xorlab.setup_torch(args.seed)
    rinit = quiet(rcnn.training_cnn, loader(X, y), loader(Xte, yte))
    xorlab.setup_torch(args.seed)
    ccfg = Config(arch="cnn", epochs=E, readout_epochs=E)
    mc = xorlab.CNNModel(ccfg, dev)
    init = {k: getattr(mc, k).clone() for k in ("W", "bW", "V", "bV", "B")}
    exact("CNN initialisation from the same seed",
          [(k, rinit[r], init[k]) for k, r in (("W", "W"), ("bW", "b_W"), ("V", "V"), ("bV", "b_V"), ("B", "B"))])
    rcnn.epochs = E
    xorlab.setup_torch(args.seed)
    ref_cnn = quiet(rcnn.training_cnn, loader(X, y), loader(Xte, yte))

    C0 = torch.zeros(28 * 7 * 7, 10, device=dev)
    b0 = torch.zeros(10, device=dev)
    xb = mc.prep(X[:256].to(dev))

    def cnn_step(state, tag):
        W, bW, V, bV, B = (state[k].clone() for k in ("W", "bW", "V", "bV", "B"))
        # repository: forward + one update, using its own functions
        hid, hid_cl, pred, _ = rcnn.forward(xb, W, bW, V, bV, C0, b0)
        for k, v in state.items():
            setattr(mc, k, v.clone())
        hid_x, y_x = mc.forward(xb)
        pooled_x, _ = mc._enc(xb)
        exact(f"CNN forward ({tag})", [("unpooled hidden", hid, hid_x), ("pooled", hid_cl, pooled_x),
                                       ("reconstruction", pred, y_x)])
        e = xb != pred
        xs = (e & xb.bool()).float() - (e & ~xb.bool()).float()
        Vr = V + rcnn.eta_dec * rcnn.decoder_update(hid, xs, 3, 1, 28)
        bVr = bV + rcnn.eta_b_dec * xs.mean(dim=(0, 2, 3), keepdim=True)
        es = torch.sign(F.conv2d(xs, B, padding=1))
        Wr = W + rcnn.eta_enc * rcnn.encoder_update(xb, es, 3, 1, 28)
        bWr = bW + rcnn.eta_b_enc * es.mean(dim=(0, 2, 3), keepdim=True)
        # xorlab: one update from the same state
        d = xorlab.xor_error(xb, y_x, "signed")
        mc.update(xb, hid_x, y_x, d, d)
        rel_close(f"CNN single update ({tag})",
                  [("V", Vr, mc.V), ("bV", bVr, mc.bV), ("W", Wr, mc.W), ("bW", bWr, mc.bW)],
                  {"V": V, "bV": bV, "W": W, "bW": bW})

    cnn_step(init, "at initialisation")
    trained = {"W": ref_cnn["W"], "bW": ref_cnn["b_W"], "V": ref_cnn["V"], "bV": ref_cnn["b_V"], "B": ref_cnn["B"]}
    cnn_step(trained, "after repository training")

    readout_check("CNN", "CNN", xorlab.CNNModel(ccfg, dev), ref_cnn, rcnn)

    # CNN full training: reported, not a pass/fail criterion
    for k, v in init.items():
        setattr(mc, k, v.clone())
    xorlab.train_xor(mc, ccfg, X, Xte, dev, gen(), [])
    agree = []
    for i in range(0, Xte.shape[0], 256):
        xt = mc.prep(Xte[i:i + 256].to(dev))
        _, _, pr, _ = rcnn.forward(xt, ref_cnn["W"], ref_cnn["b_W"], ref_cnn["V"], ref_cnn["b_V"], C0, b0)
        _, px = mc.forward(xt)
        agree.append((pr == px).float().mean().item())
    agr = sum(agree) / len(agree)
    dW = (mc.W - ref_cnn["W"]).abs().max().item() / ref_cnn["W"].abs().max().item()
    dV = (mc.V - ref_cnn["V"]).abs().max().item() / ref_cnn["V"].abs().max().item()
    report(f"CNN training, {E} epochs x {n} images (reported)", agr > 0.999,
           f"test reconstructions agree on {100 * agr:.3f}% of pixels; max rel. |ΔW| {dW:.2g}, |ΔV| {dV:.2g}",
           soft=True)

    fails = sum(t == "FAIL" for t, _ in RESULTS)
    print("\nPASS" if not fails else f"\nFAIL ({fails} check{'s' if fails > 1 else ''})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
