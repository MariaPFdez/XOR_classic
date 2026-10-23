#!/usr/bin/env python3
"""
Print every number quoted in manuscript v3 (Tables 1-3 and text) from results/,
with the configuration it comes from, so the paper can be checked against the runs.

    python paper_numbers.py [--out results]
"""
import argparse
import glob
import json
import os

import numpy as np


def load(out, cfg):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, cfg, 'seed*.json')))]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    out = ap.parse_args().out
    L = lambda c: load(out, c)

    def recon(cfg, ds, lp=True):
        R = L(cfg)
        print(f'  {cfg:28s} {ds:6s} bit {ms(R, f"recon_after/{ds}/bit_acc", 100)} | '
              f'bal {ms(R, f"recon_after/{ds}/balanced_acc", 100)} | '
              f'F1 {ms(R, f"recon_after/{ds}/f1_pos", 1, 3)}'
              + (f' | LPIPS {ms(R, f"recon_after/{ds}/lpips_fixed", 1, 3)}' if lp else ''))

    print('Table 1 (fully connected, A_fc_ref)')
    for ds in ('MNIST', 'EMNIST'):
        recon('A_fc_ref', ds)
        print(f'  all-zeros {ds}: {ms(L("A_fc_ref"), f"recon_after/{ds}/allzero_bit_acc", 100)}')

    print('\nTable 2 (convolutional)')
    for ds in ('MNIST', 'EMNIST'):
        recon('A_cnn_ref', ds)
    for ds in ('MNIST', 'EMNIST'):
        recon('F_cnn_nearest', ds, lp=False)

    print('\nTable 3 (controls): bal MNIST | bal Random | readout MNIST perc | MNIST LS | EMNIST LS | active')
    rows = ['A_fc_ref', 'C_fc_frozen_enc', 'G_fc_unsigned',
            'A_cnn_ref', 'C_cnn_frozen_enc', 'K_cnn_negb_frozen', 'K_cnn_negb_trained',
            'F_cnn_nearest', 'F_cnn_nearest_frozen', 'K_cnn_nearest_negb_frozen', 'K_cnn_nearest_negb_trained']
    for c in rows:
        R = L(c)
        print(f'  {c:28s}', ms(R, 'recon_after/MNIST/balanced_acc', 100), '|',
              ms(R, 'recon_after/Random/balanced_acc', 100), '|',
              ms(R, 'readout/MNIST_trained/perceptron_acc', 100), '|',
              ms(R, 'readout/MNIST_trained/ridge_acc', 100), '|',
              ms(R, 'readout/EMNIST_trained/ridge_acc', 100), '|',
              ms(R, 'readout/MNIST_trained/active_frac', 1, 3))
    R = L('A_fc_ref')
    print(f'  {"raw pixels":28s}', 'perc', ms(R, 'readout/MNIST_raw_pixels/perceptron_acc', 100),
          '| MNIST LS', ms(R, 'readout/MNIST_raw_pixels/ridge_acc', 100),
          '| EMNIST LS', ms(R, 'readout/EMNIST_raw_pixels/ridge_acc', 100))
    print(f'  {"CNN initial encoder":28s}', 'active', ms(L('A_cnn_ref'), 'readout/MNIST_init_encoder/active_frac', 1, 3))

    print('\nText, Section 3.1 (repeated plasticity steps; residual errors)')
    for c in ('A_fc_ref', 'B_fc_corr1', 'B_fc_corr1_ep15', 'A_cnn_ref', 'B_cnn_corr1'):
        R = L(c)
        print(f'  {c:28s} test bit {ms(R, "recon_after/MNIST/bit_acc", 100)}  updates {ms(R, "weight_updates", 1, 0)}')
    print('  A_fc_ref Hamming/image', ms(L('A_fc_ref'), 'recon_after/MNIST/hamming_per_image', 1, 1))

    print('\nText, readout')
    R = L('A_fc_ref')
    print('  FC perceptron MNIST', ms(R, 'readout/MNIST_trained/perceptron_acc', 100),
          '| EMNIST', ms(R, 'readout/EMNIST_trained/perceptron_acc', 100))
    print('  raw pixels perceptron MNIST', ms(R, 'readout/MNIST_raw_pixels/perceptron_acc', 100),
          '| EMNIST', ms(R, 'readout/EMNIST_raw_pixels/perceptron_acc', 100))
    worst = [100 * min(get(r, 'readout/MNIST_trained/per_class')) for r in R if get(r, 'readout/MNIST_trained/per_class')]
    print('  FC worst digit per seed:', ' '.join(f'{w:.1f}' for w in worst))
    R = L('A_cnn_ref')
    print('  CNN perceptron MNIST', ms(R, 'readout/MNIST_trained/perceptron_acc', 100),
          '| EMNIST', ms(R, 'readout/EMNIST_trained/perceptron_acc', 100))
    print('  CNN nearest perceptron MNIST', ms(L('F_cnn_nearest'), 'readout/MNIST_trained/perceptron_acc', 100))

    print('\nText, Section 3.3 (bottleneck, random-image training)')
    print('  FC trained on Random -> MNIST bal', ms(L('E_fc_randomtrain'), 'recon_after/MNIST/balanced_acc', 100))
    for h in ('0064', '0128', '0256', '0512', '2048'):
        Rt, Rf = L(f'D_fc_h{h}'), L(f'D_fc_frozen_h{h}')
        print(f'  H={int(h):4d} bal trained {ms(Rt, "recon_after/MNIST/balanced_acc", 100)} | fixed '
              f'{ms(Rf, "recon_after/MNIST/balanced_acc", 100)} | LS trained '
              f'{ms(Rt, "readout/MNIST_trained/ridge_acc", 100)} | LS fixed {ms(Rf, "readout/MNIST_trained/ridge_acc", 100)}')


if __name__ == '__main__':
    main()
