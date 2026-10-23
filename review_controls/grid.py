#!/usr/bin/env python3
"""
Experiment grid for the Neurocomputing revision.  Resumable and shardable:
each (config, seed) writes results/<config>/seed<k>.json; finished runs are
skipped.  Shard k of n takes every n-th job, so three DGX nodes sharing the
same NFS directory can run disjoint parts of the grid.

  python grid.py --list
  python grid.py --shard 0 --nshards 3            # on node 0, etc.
  python grid.py --only 'A_*' --seeds 0           # quick preview
"""
import argparse
import fnmatch
import json
import os
import socket
import sys
import time
import traceback
from dataclasses import replace

import torch

import xorlab
from xorlab import Config

# ----------------------------------------------------------------------------
# grid: name -> (Config overrides, run options).  Each entry answers a
# specific referee comment (see README.md).
# ----------------------------------------------------------------------------
FC = dict(arch="fc")
CNN = dict(arch="cnn")

EXPS = {
    # A. reference reproduction (comment 2: variance over seeds; 8: fixed LPIPS)
    "A_fc_ref":          (dict(**FC, lpips=True, save_weights=True), dict(raw_baseline=True)),
    "A_cnn_ref":         (dict(**CNN, lpips=True, save_weights=True), {}),
    # B. what the "iterative corrections" really are (comment 1)
    "B_fc_corr1":        (dict(**FC, max_corrections=1), {}),
    "B_fc_corr1_ep15":   (dict(**FC, max_corrections=1, epochs=15), {}),   # matched #updates
    "B_cnn_corr1":       (dict(**CNN, max_corrections=1), {}),
    "B_cnn_corr1_ep15":  (dict(**CNN, max_corrections=1, epochs=15), {}),
    # C. decoder-only learning with a fixed random encoder (comment 5a)
    "C_fc_frozen_enc":   (dict(**FC, freeze_encoder=True), {}),
    "C_cnn_frozen_enc":  (dict(**CNN, freeze_encoder=True), {}),
    # D. bottleneck sweep (comment 4); H=1024 is A_fc_ref
    **{f"D_fc_h{h:04d}": (dict(**FC, hidden=h), {}) for h in (64, 128, 256, 512, 2048)},
    **{f"D_fc_frozen_h{h:04d}": (dict(**FC, hidden=h, freeze_encoder=True), {}) for h in (64, 128, 256, 512, 2048)},
    # E. train on matched-density random images (comment 4: identity vs structure)
    "E_fc_randomtrain":       (dict(**FC, train_data="Random"), {}),
    "E_fc_randomtrain_h0256": (dict(**FC, train_data="Random", hidden=256), {}),
    "E_cnn_randomtrain":      (dict(**CNN, train_data="Random"), {}),
    # F. CNN without the max-unpool index side channel (comment 6)
    "F_cnn_nearest":          (dict(**CNN, upsample="nearest"), {}),
    "F_cnn_nearest_frozen":   (dict(**CNN, upsample="nearest", freeze_encoder=True), {}),
    # G. unsigned XOR control (comment 5d; v1 text)
    "G_fc_unsigned":     (dict(**FC, error="unsigned"), {}),
    "G_cnn_unsigned":    (dict(**CNN, error="unsigned"), {}),
    # H. graded (non-sign) random feedback to the encoder (comment 3)
    "H_fc_raw_feedback": (dict(**FC, enc_feedback="raw"), {}),
    # I. bias learning off (comment 9: summed weights vs averaged biases)
    "I_fc_no_bias":      (dict(**FC, bias_mode="none"), {}),
    "I_cnn_no_bias":     (dict(**CNN, bias_mode="none"), {}),
    # J. backprop-through-STE upper bound, same architectures (comment 5c)
    "J_fc_ste":          (dict(**FC, rule="ste"), {}),
    "J_cnn_ste":         (dict(**CNN, rule="ste"), {}),
    "J_cnn_ste_nearest": (dict(**CNN, rule="ste", upsample="nearest"), {}),
    # K. CNN init degeneracy: Theta(0)=1 with b=0 makes all-background patches active
    "K_cnn_negb_frozen":          (dict(**CNN, enc_bias_init=-1e-3, freeze_encoder=True), {}),
    "K_cnn_negb_trained":         (dict(**CNN, enc_bias_init=-1e-3), {}),
    "K_cnn_bias_only":            (dict(**CNN, freeze_W=True), {}),
    "K_cnn_nearest_negb_frozen":  (dict(**CNN, upsample="nearest", enc_bias_init=-1e-3, freeze_encoder=True), {}),
    "K_cnn_nearest_negb_trained": (dict(**CNN, upsample="nearest", enc_bias_init=-1e-3), {}),
    "K_cnn_nearest_bias_only":    (dict(**CNN, upsample="nearest", freeze_W=True), {}),
}


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def jobs(seeds, pattern):
    out = []
    for seed in seeds:                       # seed-major: seed 0 of everything first
        for name, (over, opts) in EXPS.items():
            if pattern and not any(fnmatch.fnmatch(name, p) for p in pattern.split(",")):
                continue
            out.append((name, seed, over, opts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("XORLAB_DATA", "./data"))
    ap.add_argument("--out", default="./results")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--only", default="", help="comma-separated glob(s) on config names")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-lpips", action="store_true", help="skip LPIPS even in A_* runs")
    args = ap.parse_args()

    J = jobs(parse_seeds(args.seeds), args.only)
    mine = [j for i, j in enumerate(J) if i % args.nshards == args.shard]

    if args.list:
        for i, (n, s, o, _) in enumerate(J):
            mark = "*" if i % args.nshards == args.shard else " "
            print(f"{mark} {i:3d} {n:24s} seed{s}  {o}")
        print(f"{len(J)} jobs total, {len(mine)} in shard {args.shard}/{args.nshards}")
        return

    host = socket.gethostname()
    dev = "cuda:" + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[{host}] shard {args.shard}/{args.nshards}: {len(mine)} jobs on {dev}", flush=True)

    for k, (name, seed, over, opts) in enumerate(mine):
        path = os.path.join(args.out, name, f"seed{seed}.json")
        if os.path.exists(path):
            print(f"  skip {name} seed{seed} (done)", flush=True)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cfg = replace(Config(), name=name, seed=seed, **over)
        if args.no_lpips:
            cfg = replace(cfg, lpips=False)
        t0 = time.time()
        print(f"  [{k + 1}/{len(mine)}] {name} seed{seed} ...", end="", flush=True)
        try:
            res, state = xorlab.run(cfg, args.data_root, **opts)
        except Exception:
            print(" FAILED", flush=True)
            traceback.print_exc()
            with open(path.replace(".json", ".error.txt"), "w") as f:
                f.write(traceback.format_exc())
            continue
        res["host"] = host
        tmp = path + f".tmp.{host}"
        with open(tmp, "w") as f:
            json.dump(res, f, indent=1)
        os.replace(tmp, path)                          # atomic on NFS rename
        if state is not None:
            torch.save(state, path.replace(".json", ".weights.pt"))
        a = res["recon_after"]["MNIST"]
        ro = res.get("readout", {}).get("MNIST_trained", {})
        print(f" {time.time() - t0:6.1f}s  MNIST bit={a['bit_acc']:.4f} "
              f"bal={a['balanced_acc']:.4f} F1={a['f1_pos']:.3f} "
              f"readout={ro.get('perceptron_acc', float('nan')):.4f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
