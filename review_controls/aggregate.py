#!/usr/bin/env python3
"""
Aggregate results/<config>/seed*.json into
  results/summary.md   human-readable tables (mean ± s.d. over seeds), paired
                       seed-by-seed differences, ridge-λ check, environment
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


# Pairs that share initialisation and batch order for a given seed (same
# architecture, shapes and training data), so differences can be taken per seed.
PAIRS = [("C_fc_frozen_enc", "A_fc_ref"), ("G_fc_unsigned", "A_fc_ref"),
         ("H_fc_raw_feedback", "A_fc_ref"), ("H_fc_raw_feedback_norm", "A_fc_ref"),
         ("I_fc_no_bias", "A_fc_ref"), ("B_fc_corr1_ep15", "A_fc_ref"),
         ("C_cnn_frozen_enc", "A_cnn_ref"), ("K_cnn_negb_frozen", "A_cnn_ref"),
         ("K_cnn_negb_trained", "K_cnn_negb_frozen"), ("F_cnn_nearest", "A_cnn_ref"),
         ("F_cnn_nearest_frozen", "F_cnn_nearest"),
         ("K_cnn_nearest_negb_trained", "K_cnn_nearest_negb_frozen"),
         ("B_cnn_corr1_ep15", "A_cnn_ref")] + \
        [(f"D_fc_frozen_h{h:04d}", f"D_fc_h{h:04d}") for h in (64, 128, 256, 512, 2048)]
PAIR_METRICS = [("MNIST bal. acc. (%)", "recon_after/MNIST/balanced_acc", 100),
                ("EMNIST bal. acc. (%)", "recon_after/EMNIST/balanced_acc", 100),
                ("Random bal. acc. (%)", "recon_after/Random/balanced_acc", 100),
                ("MNIST perceptron (%)", "readout/MNIST_trained/perceptron_acc", 100),
                ("MNIST ridge (%)", "readout/MNIST_trained/ridge_acc", 100),
                ("EMNIST ridge (%)", "readout/EMNIST_trained/ridge_acc", 100)]


def paired(Ra, Rb, path, scale=1.0):
    """Per-seed difference a - b over the seeds present in both."""
    va = {r["config"]["seed"]: get(r, path) for r in Ra}
    vb = {r["config"]["seed"]: get(r, path) for r in Rb}
    seeds = sorted(s for s in va if s in vb and va[s] is not None and vb[s] is not None)
    if not seeds:
        return None
    d = np.array([va[s] - vb[s] for s in seeds], dtype=float) * scale
    return d.mean(), (d.std(ddof=1) if d.size > 1 else 0.0), int((d > 0).sum()), d.size


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
            scal += [f"readout/{ds}_{tag}/perceptron_acc", f"readout/{ds}_{tag}/ridge_acc",
                     f"readout/{ds}_{tag}/ridge_lambda"]
    scal += ["weight_updates", "n_params", "wall_sec"]
    for n in names:
        for p in scal:
            t = ms(R[n], p)
            if t:
                csv.append(f"{n},{p},{t[0]:.6g},{t[1]:.6g},{t[2]}")

    md.append("# XOR revision experiments — summary\n")
    md.append("Mean ± s.d. over seeds (n given). All reconstruction metrics on test sets, "
              "single forward pass (as in `model_accuracy`).\n")

    # ---------- environment ----------
    env = defaultdict(int)
    for n in names:
        for r in R[n]:
            env[(r.get("torch", "?"), r.get("device", "?"), r.get("host", "?"))] += 1
    md.append("\n## Environment\n\n" + md_table(["torch", "device", "host", "runs"],
                                                 [list(k) + [v] for k, v in sorted(env.items())]))

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
        rows.append(["raw pixels (784)",
                     fmt(ms(raw, "readout/MNIST_raw_pixels/perceptron_acc", 100)),
                     fmt(ms(raw, "readout/MNIST_raw_pixels/ridge_acc", 100)), "", "",
                     fmt(ms(raw, "readout/EMNIST_raw_pixels/perceptron_acc", 100)),
                     fmt(ms(raw, "readout/EMNIST_raw_pixels/ridge_acc", 100)), "", ""])
    md.append("\n## Linear readout accuracy (%) — perceptron rule (as paper) and ridge (best linear, "
              "λ on train split)\n\n'init enc' = same run's encoder before training "
              "(random-feature baseline).\n\n"
              + md_table(hdr, [["**" + r[0] + "**"] + r[1:] if r[0] == "raw pixels (784)" else r
                               for r in rows]))
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

    # ---------- ridge lambda: is the chosen value on the edge of the grid? ----------
    LAM_EDGE = (1e-2, 1e3)
    rows = []
    for n in names:
        for key in sorted({k for r in R[n] for k in (r.get("readout") or {})}):
            lams = [get(r, f"readout/{key}/ridge_lambda") for r in R[n]]
            lams = [l for l in lams if l is not None]
            if lams:
                edge = sum(l in LAM_EDGE for l in lams)
                rows.append([n, key, " ".join(f"{l:g}" for l in lams),
                             f"**{edge}/{len(lams)}**" if edge else "0"])
    if rows:
        md.append("\n## Ridge λ chosen per seed (grid 1e-2 … 1e3; 'edge' = runs at the boundary)\n\n"
                  + md_table(["config", "readout", "λ per seed", "edge"], rows))

    # ---------- paired, seed-by-seed differences ----------
    rows = []
    for a, b in PAIRS:
        if a not in R or b not in R:
            continue
        for label, path, scale in PAIR_METRICS:
            d = paired(R[a], R[b], path, scale)
            if d is not None:
                rows.append([a, b, label, f"{d[0]:+.2f} ± {d[1]:.2f}", f"{d[2]}/{d[3]}"])
    if rows:
        md.append("\n## Paired differences (A − B), matched by seed\n\n"
                  "Mean ± s.d. of the per-seed difference; 'A > B' counts seeds where A is higher.\n\n"
                  + md_table(["A", "B", "metric", "A − B", "A > B"], rows))

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
            sd = v.std(0, ddof=1) if v.shape[0] > 1 else np.zeros(L)
            ax[j].fill_between(x, v.mean(0) - sd, v.mean(0) + sd, alpha=.2)
    ax[0].set_title("train bit acc. (online, pre-update)"); ax[1].set_title("MNIST test bit acc.")
    for a in ax:
        a.set_xlabel("epoch"); a.set_ylabel("%"); a.grid(alpha=.3)
    ax[1].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "learning_curves.pdf")); plt.close(fig)

    # bottleneck sweep: trained encoder (A_fc_ref, D_fc_h*) vs fixed random encoder
    # (C_fc_frozen_enc, D_fc_frozen_h*); H = 1024 is the reference width
    def sweep(prefix, ref):
        pts = [(R[n][0]["config"]["hidden"], n) for n in R if n.startswith(prefix) or n == ref]
        return sorted(pts)

    def series(ax, pts, path, **kw):
        m = [(h, ms(R[n], path, 100)) for h, n in pts]
        m = [(h, t) for h, t in m if t is not None]
        if m:
            ax.errorbar([h for h, _ in m], [t[0] for _, t in m], [t[1] for _, t in m], capsize=3, **kw)

    trained, frozen = sweep("D_fc_h", "A_fc_ref"), sweep("D_fc_frozen_h", "C_fc_frozen_enc")
    if trained or frozen:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for ci, (ds, mk) in enumerate((("MNIST", "o"), ("EMNIST", "s"), ("Random", "^"))):
            col = f"C{ci}"
            series(ax[0], trained, f"recon_after/{ds}/balanced_acc", marker=mk, c=col, label=f"{ds} trained")
            series(ax[0], frozen, f"recon_after/{ds}/balanced_acc", marker=mk, c=col, ls="--", mfc="none",
                   label=f"{ds} fixed")
        series(ax[1], trained, "readout/MNIST_trained/perceptron_acc", marker="o", c="C0", label="trained encoder")
        series(ax[1], frozen, "readout/MNIST_trained/perceptron_acc", marker="o", c="C2", ls="--", mfc="none",
               label="fixed encoder")
        raw = [r for n in R for r in R[n] if get(r, "readout/MNIST_raw_pixels")]
        if raw:
            t = ms(raw, "readout/MNIST_raw_pixels/perceptron_acc", 100)
            ax[1].axhline(t[0], ls=":", c="k", lw=1, label="raw pixels")
        for a in ax:
            a.set_xscale("log", base=2); a.set_xlabel("hidden units"); a.grid(alpha=.3); a.legend(fontsize=7)
            a.axvline(784, c="grey", lw=.8, ls=":")
        ax[0].set_ylabel("balanced recon. acc. (%)"); ax[1].set_ylabel("MNIST perceptron readout (%)")
        fig.tight_layout(); fig.savefig(os.path.join(figdir, "bottleneck_sweep.pdf")); plt.close(fig)

    # per-class spread across seeds
    for n in ("A_fc_ref", "A_cnn_ref"):
        P = [(r["config"]["seed"], get(r, "readout/MNIST_trained/per_class")) for r in R.get(n, [])]
        P = sorted((s, np.array(p) * 100) for s, p in P if p)
        if P:
            fig, ax = plt.subplots(figsize=(6, 3.4))
            for s, p in P:
                ax.plot(range(10), p, marker="o", ms=3, alpha=.7, label=f"seed {s}")
            ax.set_xticks(range(10)); ax.set_xlabel("digit"); ax.set_ylabel("readout acc. (%)")
            ax.set_title(f"{n}: per-class readout across seeds"); ax.grid(alpha=.3); ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(os.path.join(figdir, f"per_class_{n}.pdf")); plt.close(fig)


if __name__ == "__main__":
    main()
