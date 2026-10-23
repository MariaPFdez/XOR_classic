#!/usr/bin/env python3
"""
Aggregate results/<config>/seed*.json into
  results/summary.md   human-readable tables (mean ± s.d. over seeds)
  results/summary.csv  long format, one row per (config, metric)
  results/tables.tex   booktabs tables ready to paste
  results/figs/*.pdf   learning curves, bottleneck sweep, per-class spread
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def load(out):
    runs = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(out, "*", "seed*.json"))):
        with open(p) as f:
            runs[os.path.basename(os.path.dirname(p))].append(json.load(f))
    return runs


def get(r, path):
    cur = r
    for k in path.split("/"):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def ms(runs, path, scale=1.0):
    v = [get(r, path) for r in runs]
    v = np.array([x for x in v if x is not None], dtype=float) * scale
    if v.size == 0:
        return None
    return v.mean(), (v.std(ddof=1) if v.size > 1 else 0.0), v.size


def fmt(t, d=2):
    if t is None:
        return "–"
    m, s, n = t
    return f"{m:.{d}f} ± {s:.{d}f}" if n > 1 else f"{m:.{d}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def tex_table(caption, header, rows):
    cols = "l" + "c" * (len(header) - 1)
    body = " \\\\\n".join(" & ".join(str(c).replace("±", "$\\pm$").replace("_", "\\_")
                                      for c in r) for r in rows)
    return (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n"
            f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n"
            + " & ".join(header) + " \\\\\n\\midrule\n" + body +
            " \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


RECON_COLS = [("bit acc. (%)", "bit_acc", 100, 2), ("bal. acc. (%)", "balanced_acc", 100, 2),
              ("F1", "f1_pos", 1, 3), ("perfect img (%)", "perfect_frac", 100, 1),
              ("Hamming/img", "hamming_per_image", 1, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./results")
    args = ap.parse_args()
    R = load(args.out)
    if not R:
        raise SystemExit(f"no results under {args.out}")
    names = sorted(R)
    md, tex, csv = [], [], ["config,metric,mean,sd,n"]

    # ---------- long csv of every scalar we know about ----------
    scal = []
    for ds in ("MNIST", "EMNIST", "Random"):
        for _, k, _, _ in RECON_COLS:
            scal.append(f"recon_after/{ds}/{k}")
        scal += [f"recon_after/{ds}/allzero_bit_acc", f"recon_after/{ds}/lpips_fixed",
                 f"recon_after/{ds}/lpips_repo_style", f"recon_before/{ds}/bit_acc"]
    for ds in ("MNIST", "EMNIST"):
        for tag in ("trained", "init_encoder", "raw_pixels"):
            scal += [f"readout/{ds}_{tag}/perceptron_acc", f"readout/{ds}_{tag}/ridge_acc"]
    scal += ["weight_updates", "n_params", "wall_sec"]
    for n in names:
        for p in scal:
            t = ms(R[n], p)
            if t:
                csv.append(f"{n},{p},{t[0]:.6g},{t[1]:.6g},{t[2]}")

    md.append("# XOR revision experiments — summary\n")
    md.append("Mean ± s.d. over seeds (n given). All reconstruction metrics on test sets, "
              "single forward pass (as in `model_accuracy`).\n")

    # ---------- T1 reconstruction ----------
    for ds in ("MNIST", "EMNIST", "Random"):
        rows = []
        for n in names:
            rows.append([n, ms(R[n], "weight_updates")[2]]
                        + [fmt(ms(R[n], f"recon_after/{ds}/{k}", s), d) for _, k, s, d in RECON_COLS]
                        + [fmt(ms(R[n], f"recon_after/{ds}/allzero_bit_acc", 100))])
        hdr = ["config", "n"] + [c for c, *_ in RECON_COLS] + ["all-zeros (%)"]
        md.append(f"\n## Reconstruction — {ds} test\n\n" + md_table(hdr, rows))
    main_rows = [[n] + [fmt(ms(R[n], f"recon_after/MNIST/{k}", s), d) for _, k, s, d in RECON_COLS[:3]]
                 + [fmt(ms(R[n], f"recon_after/EMNIST/{k}", s), d) for _, k, s, d in RECON_COLS[:3]]
                 for n in names]
    tex.append(tex_table("Reconstruction on MNIST and EMNIST test sets (mean $\\pm$ s.d. over seeds).",
                         ["Model", "MNIST bit", "MNIST bal.", "MNIST F1",
                          "EMNIST bit", "EMNIST bal.", "EMNIST F1"], main_rows))

    # ---------- LPIPS ----------
    rows = []
    for n in names:
        for ds in ("MNIST", "EMNIST"):
            a, b = ms(R[n], f"recon_after/{ds}/lpips_fixed"), ms(R[n], f"recon_after/{ds}/lpips_repo_style")
            if a or b:
                rows.append([n, ds, fmt(a, 3), fmt(b, 3)])
    if rows:
        md.append("\n## LPIPS: corrected (all images, normalize=True) vs repo-style "
                  "(first image per 256-batch, un-normalised)\n\n"
                  + md_table(["config", "data", "LPIPS fixed", "LPIPS repo-style"], rows))

    # ---------- T2 readout ----------
    rows = []
    for n in names:
        row = [n]
        for ds in ("MNIST", "EMNIST"):
            for tag in ("trained", "init_encoder"):
                row += [fmt(ms(R[n], f"readout/{ds}_{tag}/perceptron_acc", 100)),
                        fmt(ms(R[n], f"readout/{ds}_{tag}/ridge_acc", 100))]
        rows.append(row)
    raw = [r for n in names for r in R[n] if get(r, "readout/MNIST_raw_pixels")]
    hdr = ["config", "MNIST perc.", "MNIST ridge", "MNIST perc. (init enc)", "MNIST ridge (init enc)",
           "EMNIST perc.", "EMNIST ridge", "EMNIST perc. (init enc)", "EMNIST ridge (init enc)"]
    if raw:
        rows.append(["**raw pixels (784)**",
                     fmt(ms(raw, "readout/MNIST_raw_pixels/perceptron_acc", 100)),
                     fmt(ms(raw, "readout/MNIST_raw_pixels/ridge_acc", 100)), "", "",
                     fmt(ms(raw, "readout/EMNIST_raw_pixels/perceptron_acc", 100)),
                     fmt(ms(raw, "readout/EMNIST_raw_pixels/ridge_acc", 100)), "", ""])
    md.append("\n## Linear readout accuracy (%) — perceptron rule (as paper) and ridge (best linear, "
              "λ on train split)\n\n'init enc' = same run's encoder before training "
              "(random-feature baseline).\n\n" + md_table(hdr, rows))
    tex.append(tex_table("Linear readout accuracy (\\%) of latent codes; init = untrained encoder.",
                         ["Model", "MNIST", "MNIST (init)", "EMNIST", "EMNIST (init)"],
                         [[r[0], r[1], r[3], r[5], r[7]] for r in rows]))

    # ---------- T3 per-class spread ----------
    rows = []
    for n in names:
        for r in R[n]:
            pc = get(r, "readout/MNIST_trained/per_class")
            if pc and n.startswith("A_"):
                pc = np.array(pc) * 100
                rows.append([n, r["config"]["seed"], f"{pc.min():.1f}", f"{pc.max():.1f}",
                             int(pc.argmin()), " ".join(f"{v:.0f}" for v in pc)])
    if rows:
        md.append("\n## Per-class MNIST readout accuracy by seed (reference models)\n\n"
                  + md_table(["config", "seed", "min", "max", "worst digit", "per-class (0..9)"], rows))

    # ---------- T4 corrections ----------
    rows = []
    for n in names:
        if n.startswith(("A_", "B_")):
            last = [r["log"][-1] for r in R[n] if r.get("log")]
            pre = ms([{"v": l.get("train_acc_pre_update")} for l in last], "v", 100)
            post = ms([{"v": l.get("train_acc_post_corrections")} for l in last], "v", 100)
            rows.append([n, fmt(ms(R[n], "weight_updates"), 0), fmt(pre), fmt(post),
                         fmt(ms(R[n], "recon_after/MNIST/bit_acc", 100))])
    if rows:
        md.append("\n## Corrections = repeated plasticity per mini-batch\n\n"
                  "train(pre) = online accuracy before updating on the batch; train(post) = repo's "
                  "printed 'Accuracy', measured after up to max_corrections updates on that same batch.\n\n"
                  + md_table(["config", "weight updates", "last-epoch train (pre) %",
                              "last-epoch train (post) %", "MNIST test bit %"], rows))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(args.out, "summary.csv"), "w") as f:
        f.write("\n".join(csv) + "\n")
    with open(os.path.join(args.out, "tables.tex"), "w") as f:
        f.write("% generated by aggregate.py\n" + "\n".join(tex))

    figures(R, os.path.join(args.out, "figs"))
    print(f"wrote summary.md, summary.csv, tables.tex, figs/ in {args.out}")


def figures(R, figdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figures")
        return
    os.makedirs(figdir, exist_ok=True)

    # learning curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for n in [k for k in sorted(R) if k.startswith(("A_", "B_", "C_", "F_", "G_"))]:
        logs = [r["log"] for r in R[n] if r.get("log") and "test_bit_acc" in r["log"][0]]
        if not logs:
            continue
        L = min(len(l) for l in logs)
        x = np.arange(1, L + 1)
        for j, key in enumerate(("train_acc_pre_update", "test_bit_acc")):
            v = np.array([[l[e][key] for e in range(L)] for l in logs]) * 100
            ax[j].plot(x, v.mean(0), marker="o", ms=3, label=n)
            ax[j].fill_between(x, v.mean(0) - v.std(0), v.mean(0) + v.std(0), alpha=.2)
    ax[0].set_title("train bit acc. (online, pre-update)"); ax[1].set_title("MNIST test bit acc.")
    for a in ax:
        a.set_xlabel("epoch"); a.set_ylabel("%"); a.grid(alpha=.3)
    ax[1].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "learning_curves.pdf")); plt.close(fig)

    # bottleneck sweep
    pts = []
    for n in R:
        if n.startswith("D_fc_h") or n == "A_fc_ref":
            h = R[n][0]["config"]["hidden"]
            pts.append((h, n))
    if pts:
        pts.sort()
        H = [p[0] for p in pts]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for ds, mk in (("MNIST", "o"), ("EMNIST", "s"), ("Random", "^")):
            m = [ms(R[n], f"recon_after/{ds}/balanced_acc", 100) for _, n in pts]
            ax[0].errorbar(H, [t[0] for t in m], [t[1] for t in m], marker=mk, label=ds, capsize=3)
        for tag, mk in (("trained", "o"), ("init_encoder", "x")):
            m = [ms(R[n], f"readout/MNIST_{tag}/perceptron_acc", 100) for _, n in pts]
            if all(m):
                ax[1].errorbar(H, [t[0] for t in m], [t[1] for t in m], marker=mk, label=tag, capsize=3)
        raw = [r for n in R for r in R[n] if get(r, "readout/MNIST_raw_pixels")]
        if raw:
            t = ms(raw, "readout/MNIST_raw_pixels/perceptron_acc", 100)
            ax[1].axhline(t[0], ls="--", c="k", lw=1, label="raw pixels")
        for a in ax:
            a.set_xscale("log", base=2); a.set_xlabel("hidden units"); a.grid(alpha=.3); a.legend(fontsize=8)
            a.axvline(784, c="grey", lw=.8, ls=":")
        ax[0].set_ylabel("balanced recon. acc. (%)"); ax[1].set_ylabel("MNIST readout acc. (%)")
        fig.tight_layout(); fig.savefig(os.path.join(figdir, "bottleneck_sweep.pdf")); plt.close(fig)

    # per-class spread across seeds
    for n in ("A_fc_ref", "A_cnn_ref"):
        P = [get(r, "readout/MNIST_trained/per_class") for r in R.get(n, [])]
        P = np.array([p for p in P if p]) * 100
        if P.size:
            fig, ax = plt.subplots(figsize=(6, 3.4))
            for s, p in enumerate(P):
                ax.plot(range(10), p, marker="o", ms=3, alpha=.7, label=f"seed {s}")
            ax.set_xticks(range(10)); ax.set_xlabel("digit"); ax.set_ylabel("readout acc. (%)")
            ax.set_title(f"{n}: per-class readout across seeds"); ax.grid(alpha=.3); ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(os.path.join(figdir, f"per_class_{n}.pdf")); plt.close(fig)


if __name__ == "__main__":
    main()
