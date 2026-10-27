#!/usr/bin/env python3
"""
Print every number quoted in manuscript v3 (Tables 1-3, Fig. 6 and text) from
results/, with the configuration it comes from, so the paper can be checked
against the runs.  Sections follow the manuscript.

    python paper_numbers.py [--out results]
"""
import argparse
import glob
import json
import os

import numpy as np


def load(out, cfg):
    runs = []
    for p in sorted(glob.glob(os.path.join(out, cfg, 'seed*.json'))):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def get(r, path):
    for k in path.split('/'):
        if not isinstance(r, dict) or k not in r:
            return None
        r = r[k]
    return r


def ms(R, path, scale=1.0, d=2):
    if not R:
        return 'MISSING'
    v = [get(r, path) for r in R]
    v = np.array([x for x in v if x is not None], dtype=float) * scale
    if v.size == 0:
        return 'n/a'
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f'{v.mean():.{d}f} ± {sd:.{d}f}  (n={len(v)})'


def paired(Ra, Rb, path, scale=100.0):
    """a - b per seed: mean ± sd, and how many seeds have a > b."""
    va = {r['config']['seed']: get(r, path) for r in Ra}
    vb = {r['config']['seed']: get(r, path) for r in Rb}
    s = [k for k in va if k in vb and va[k] is not None and vb[k] is not None]
    if not s:
        return 'MISSING'
    d = np.array([va[k] - vb[k] for k in s]) * scale
    return f'{d.mean():+.2f} ± {d.std(ddof=1) if d.size > 1 else 0:.2f}  ({int((d > 0).sum())}/{d.size} seeds a > b)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    out = ap.parse_args().out
    cache = {}

    def L(c):
        if c not in cache:
            cache[c] = load(out, c)
        return cache[c]

    BAL = 'recon_after/{}/balanced_acc'

    def recon(cfg, ds, phase='recon_after', lp=True):
        R = L(cfg)
        print(f'  {cfg:28s} {phase[6:]:6s} {ds:6s} bit {ms(R, f"{phase}/{ds}/bit_acc", 100)} | '
              f'bal {ms(R, f"{phase}/{ds}/balanced_acc", 100)} | '
              f'F1 {ms(R, f"{phase}/{ds}/f1_pos", 1, 3)}'
              + (f' | LPIPS {ms(R, f"{phase}/{ds}/lpips_fixed", 1, 3)}' if lp else ''))

    # ------------------------------------------------------------------ methods
    print('Section 2 (Methods)')
    R = L('A_fc_ref')
    print('  parameters FC', R[0]['n_params'] if R else 'MISSING', '| CNN', L('A_cnn_ref')[0]['n_params'] if L('A_cnn_ref') else 'MISSING')
    print('  weight updates in 5 epochs (A_fc_ref)', ms(R, 'weight_updates', 1, 0))
    for ds in ('MNIST', 'EMNIST'):
        print(f'  all-zeros bit acc {ds} {ms(R, f"recon_after/{ds}/allzero_bit_acc", 100)}  '
              f'(active pixels = 100 - this)')
    print('  STE selected lr: see "STE" below; ridge λ grid extension: ridge_recheck.py')

    # ------------------------------------------------------------------ table 1
    print('\nTable 1 (fully connected, A_fc_ref)')
    recon('A_fc_ref', 'MNIST', 'recon_before', lp=False)
    recon('A_fc_ref', 'MNIST')
    recon('A_fc_ref', 'EMNIST')

    print('\nSection 3.1 text')
    print('  Hamming errors / image (MNIST)', ms(R, 'recon_after/MNIST/hamming_per_image', 1, 1),
          '| perfect images (%)', ms(R, 'recon_after/MNIST/perfect_frac', 100, 2))
    for c in ('G_fc_unsigned', 'G_cnn_unsigned'):
        Rg = L(c)
        print(f'  {c:28s} bit {ms(Rg, "recon_after/MNIST/bit_acc", 100)} | bal {ms(Rg, BAL.format("MNIST"), 100)} '
              f'| recall {ms(Rg, "recon_after/MNIST/recall_pos", 1, 3)} | precision {ms(Rg, "recon_after/MNIST/precision_pos", 1, 3)}')
    for c in ('H_fc_raw_feedback', 'H_fc_raw_feedback_norm'):
        print(f'  {c:28s} bal {ms(L(c), BAL.format("MNIST"), 100)} | paired vs A_fc_ref '
              f'{paired(L(c), R, BAL.format("MNIST"))}')
    print(f'  {"I_fc_no_bias":28s} bal {ms(L("I_fc_no_bias"), BAL.format("MNIST"), 100)}')
    print(f'  {"I_cnn_no_bias":28s} bit {ms(L("I_cnn_no_bias"), "recon_after/MNIST/bit_acc", 100)}')
    print('  repeated updates per mini-batch (MNIST test bit acc., weight updates):')
    for c in ('A_fc_ref', 'B_fc_corr1', 'B_fc_corr1_ep15', 'A_cnn_ref', 'B_cnn_corr1', 'B_cnn_corr1_ep15'):
        Rc = L(c)
        print(f'    {c:26s} bit {ms(Rc, "recon_after/MNIST/bit_acc", 100)}  updates {ms(Rc, "weight_updates", 1, 0)}')

    # ------------------------------------------------------------------ table 2
    print('\nTable 2 (convolutional)')
    recon('A_cnn_ref', 'MNIST', 'recon_before', lp=False)
    recon('A_cnn_ref', 'MNIST')
    recon('A_cnn_ref', 'EMNIST')
    recon('F_cnn_nearest', 'MNIST', lp=False)
    recon('F_cnn_nearest', 'EMNIST', lp=False)

    print('\nSection 3.2 text')
    Rc, Rf = L('A_cnn_ref'), L('F_cnn_nearest')
    print('  recall MNIST: CNN', ms(Rc, 'recon_after/MNIST/recall_pos', 1, 3), '| FC', ms(R, 'recon_after/MNIST/recall_pos', 1, 3))
    print('  Random bal: CNN unpool', ms(Rc, BAL.format('Random'), 100), '| CNN nearest', ms(Rf, BAL.format('Random'), 100),
          '| FC', ms(R, BAL.format('Random'), 100))
    print('  nearest - unpool, MNIST bal (paired)', paired(Rf, Rc, BAL.format('MNIST')))

    # ------------------------------------------------------------------ table 3
    print('\nTable 3 (controls): bal MNIST | bal EMNIST | bal Random | perc MNIST | LS MNIST | LS EMNIST | active')
    rows = ['A_fc_ref', 'C_fc_frozen_enc', 'G_fc_unsigned', 'E_fc_randomtrain', 'J_fc_ste_lr3e-3',
            'A_cnn_ref', 'C_cnn_frozen_enc', 'K_cnn_negb_frozen', 'K_cnn_negb_trained', 'J_cnn_ste_lr1e-1',
            'F_cnn_nearest', 'F_cnn_nearest_frozen', 'K_cnn_nearest_negb_frozen', 'K_cnn_nearest_negb_trained']
    for c in rows:
        Rr = L(c)
        print(f'  {c:28s}', ' | '.join(ms(Rr, p, s, d) for p, s, d in (
            (BAL.format('MNIST'), 100, 1), (BAL.format('EMNIST'), 100, 1), (BAL.format('Random'), 100, 1),
            ('readout/MNIST_trained/perceptron_acc', 100, 1), ('readout/MNIST_trained/ridge_acc', 100, 1),
            ('readout/EMNIST_trained/ridge_acc', 100, 1), ('readout/MNIST_trained/active_frac', 1, 2))))
    print(f'  {"raw pixels (A_fc_ref)":28s}', 'perc', ms(R, 'readout/MNIST_raw_pixels/perceptron_acc', 100, 1),
          '| LS MNIST', ms(R, 'readout/MNIST_raw_pixels/ridge_acc', 100, 1),
          '| LS EMNIST', ms(R, 'readout/EMNIST_raw_pixels/ridge_acc', 100, 1))

    # ------------------------------------------------------------------ 3.3 text
    print('\nSection 3.3 text')
    Rz = L('C_fc_frozen_enc')
    for ds in ('MNIST', 'EMNIST'):
        print(f'  fixed - trained FC, {ds} bal (paired)', paired(Rz, R, BAL.format(ds)),
              '| fixed', ms(Rz, BAL.format(ds), 100))
    Re = L('E_fc_randomtrain')
    print('  trained on Random: MNIST bal', ms(Re, BAL.format('MNIST'), 100), '| EMNIST bal', ms(Re, BAL.format('EMNIST'), 100),
          '| Random bal', ms(Re, BAL.format('Random'), 100))
    Rec = L('E_cnn_randomtrain')
    print('  CNN trained on Random: MNIST bal', ms(Rec, BAL.format('MNIST'), 100), '| EMNIST bal', ms(Rec, BAL.format('EMNIST'), 100))
    print('  readout FC: perc trained', ms(R, 'readout/MNIST_trained/perceptron_acc', 100),
          '| perc initial encoder', ms(R, 'readout/MNIST_init_encoder/perceptron_acc', 100),
          '| perc raw', ms(R, 'readout/MNIST_raw_pixels/perceptron_acc', 100))
    print('  LS MNIST trained', ms(R, 'readout/MNIST_trained/ridge_acc', 100), '| LS initial', ms(R, 'readout/MNIST_init_encoder/ridge_acc', 100))
    print('  EMNIST: LS trained', ms(R, 'readout/EMNIST_trained/ridge_acc', 100), '| LS initial', ms(R, 'readout/EMNIST_init_encoder/ridge_acc', 100),
          '| perc trained', ms(R, 'readout/EMNIST_trained/perceptron_acc', 100), '| perc initial', ms(R, 'readout/EMNIST_init_encoder/perceptron_acc', 100))
    worst = [100 * min(get(r, 'readout/MNIST_trained/per_class')) for r in R if get(r, 'readout/MNIST_trained/per_class')]
    print('  FC worst class per seed:', ' '.join(f'{w:.1f}' for w in worst))
    print('  Fig. 6 bottleneck (MNIST): bal trained | bal fixed | perc trained | perc fixed | LS trained | LS fixed | LS paired fixed-trained')
    for h in ('0064', '0128', '0256', '0512', '2048'):
        Rt, Rx = L(f'D_fc_h{h}'), L(f'D_fc_frozen_h{h}')
        print(f'    H={int(h):4d}', ' | '.join(ms(x, p, 100, d) for x, p, d in (
            (Rt, BAL.format('MNIST'), 2), (Rx, BAL.format('MNIST'), 2),
            (Rt, 'readout/MNIST_trained/perceptron_acc', 1), (Rx, 'readout/MNIST_trained/perceptron_acc', 1),
            (Rt, 'readout/MNIST_trained/ridge_acc', 1), (Rx, 'readout/MNIST_trained/ridge_acc', 1))),
            '|', paired(Rx, Rt, 'readout/MNIST_trained/ridge_acc'))
    print('  Fig. 6, paired trained - fixed: perc MNIST | LS MNIST | LS EMNIST | bal Random')
    for t, x, h in [(f'D_fc_h{h}', f'D_fc_frozen_h{h}', h) for h in ('0064', '0128', '0256', '0512')] + \
                   [('A_fc_ref', 'C_fc_frozen_enc', '1024'), ('D_fc_h2048', 'D_fc_frozen_h2048', '2048')]:
        print(f'    H={int(h):4d}', ' | '.join(paired(L(t), L(x), p) for p in (
            'readout/MNIST_trained/perceptron_acc', 'readout/MNIST_trained/ridge_acc',
            'readout/EMNIST_trained/ridge_acc', BAL.format('Random'))))
    Rk, Rkt = L('K_cnn_negb_frozen'), L('K_cnn_negb_trained')
    print('  CNN readout perc: trained', ms(Rc, 'readout/MNIST_trained/perceptron_acc', 100),
          '| initial (b=0)', ms(Rc, 'readout/MNIST_init_encoder/perceptron_acc', 100),
          '| active initial', ms(Rc, 'readout/MNIST_init_encoder/active_frac', 1, 3))
    print('  CNN fixed b=-1e-3: bal', ms(Rk, BAL.format('MNIST'), 100), '| perc', ms(Rk, 'readout/MNIST_trained/perceptron_acc', 100),
          '| LS', ms(Rk, 'readout/MNIST_trained/ridge_acc', 100), '| LS EMNIST', ms(Rk, 'readout/EMNIST_trained/ridge_acc', 100),
          '| active', ms(Rk, 'readout/MNIST_trained/active_frac', 1, 3))
    print('  CNN trained from b=-1e-3: bal', ms(Rkt, BAL.format('MNIST'), 100), '| perc', ms(Rkt, 'readout/MNIST_trained/perceptron_acc', 100),
          '| paired bal vs fixed', paired(Rkt, Rk, BAL.format('MNIST')))
    Rn, Rnt = L('K_cnn_nearest_negb_frozen'), L('K_cnn_nearest_negb_trained')
    print('  CNN nearest, trained vs fixed (b=-1e-3): bal', ms(Rnt, BAL.format('MNIST'), 100), 'vs', ms(Rn, BAL.format('MNIST'), 100),
          '| paired', paired(Rnt, Rn, BAL.format('MNIST')), '| LS', ms(Rnt, 'readout/MNIST_trained/ridge_acc', 100), 'vs',
          ms(Rn, 'readout/MNIST_trained/ridge_acc', 100), '| LS paired', paired(Rnt, Rn, 'readout/MNIST_trained/ridge_acc'))

    # ------------------------------------------------------------------ STE
    print('\nSTE (Methods 2.5, Table 3, Section 3.3)')
    for arch in ('fc', 'cnn'):
        print(f'  STE {arch}: lr chosen by final-epoch training loss (no test data)')
        best, all_exact = None, True
        for lr, c in (('1e-4', f'J_{arch}_ste_lr1e-4'), ('3e-4', f'J_{arch}_ste_lr3e-4'),
                      ('1e-3', f'J_{arch}_ste'), ('3e-3', f'J_{arch}_ste_lr3e-3'),
                      ('1e-2', f'J_{arch}_ste_lr1e-2'), ('3e-2', f'J_{arch}_ste_lr3e-2'),
                      ('1e-1', f'J_{arch}_ste_lr1e-1'), ('3e-1', f'J_{arch}_ste_lr3e-1')):
            Rs = L(c)
            if not Rs:
                continue
            losses = [r['log'][-1].get('loss_epoch_mean', r['log'][-1].get('loss')) for r in Rs if r.get('log')]
            exact = all('loss_epoch_mean' in r['log'][-1] for r in Rs if r.get('log'))
            all_exact = all_exact and exact
            tl = float(np.mean(losses)) if losses else float('nan')
            print(f'    lr {lr:5s} train loss {tl:.4f}{"" if exact else " (last batch only)"} | '
                  f'MNIST bal {ms(Rs, BAL.format("MNIST"), 100)} | EMNIST bal {ms(Rs, BAL.format("EMNIST"), 100)} | '
                  f'LS MNIST {ms(Rs, "readout/MNIST_trained/ridge_acc", 100, 1)} | updates {ms(Rs, "weight_updates", 1, 0)}')
            if losses and (best is None or tl < best[0]):
                best = (tl, lr)
        if best and all_exact:
            print(f'    -> selected lr {best[1]}')
        elif best:
            print(f'    -> no selection: J_{arch}_ste logs only the last-batch loss (older runs); '
                  f'delete results/J_{arch}_ste/ and rerun grid.py --only J_{arch}_ste')


if __name__ == '__main__':
    main()
