# Control experiments (`review_controls/`)

Control experiments for the manuscript *"Toward defining loss functions in neuroscience: an XOR-based neuronal mechanism for local learning"* (v3, Section 3.3 and Tables 1–3).

`xorlab.py` re-implements the training rules of the main code in this repository (`XORclassic/`). With default settings it reproduces them exactly:
- same initialisation and learning rates;
- same signed-XOR masks and sign-DFA encoder rule;
- weights updated with the batch sum, biases with the batch mean;
- `max_corrections=3`;
- CNN with max-unpooling by indices;
- the same perceptron readout.

On top of that it adds random seeds, logging and the switches needed for the controls. The CNN updates are vectorised with `einsum` but give the same values as the original per-sample loop.

## Files

| File | Purpose |
|---|---|
| `xorlab.py` | Models, training rules, metrics and readouts |
| `grid.py` | Experiment grid (37 configurations × 5 seeds = 185 runs). Resumable and shardable |
| `aggregate.py` | `results/summary.md`, `summary.csv`, `tables.tex`, `figs/` |
| `paper_numbers.py` | Prints every number quoted in the manuscript, with its source configuration |
| `check_env.py` | Checks the environment and downloads MNIST, EMNIST-balanced and the LPIPS weights once |
| `setup_env.sh`, `Dockerfile`, `launch_shards.sh` | Optional: venv, NGC container, multi-node launch |
| `results/<config>/seed<k>.json` | Raw per-run results used in the paper (weights not included) |

## Mapping to the manuscript

| Manuscript | Configurations |
|---|---|
| Table 1 | `A_fc_ref` |
| Table 2 | `A_cnn_ref` (max-unpooling), `F_cnn_nearest` (nearest-neighbour upsampling) |
| Table 3 | `A_fc_ref`, `C_fc_frozen_enc`, `G_fc_unsigned`, `A_cnn_ref`, `C_cnn_frozen_enc`, `K_cnn_negb_frozen`, `K_cnn_negb_trained`, `F_cnn_nearest`, `F_cnn_nearest_frozen`, `K_cnn_nearest_negb_frozen`, `K_cnn_nearest_negb_trained`; raw-pixel readout stored in `A_fc_ref` |
| Text: plasticity steps (Section 3.1) | `B_fc_corr1`, `B_fc_corr1_ep15`, `B_cnn_corr1` |
| Text: bottleneck, random-image training (Section 3.3) | `D_fc_h*`, `D_fc_frozen_h*`, `E_fc_randomtrain` |
| Not quoted in the text | `E_*` (others), `G_cnn_unsigned`, `H_fc_raw_feedback`, `I_*_no_bias`, `J_*_ste*` (untuned STE baseline), `K_*_bias_only` (uninformative: biases swamp the fixed small weights) |

## Reproduce

```bash
pip install torch torchvision lpips matplotlib numpy
export XORLAB_DATA=$PWD/data TORCH_HOME=$PWD/torch_home
python check_env.py                                   # must print READY
python grid.py --seeds 0-4                            # or shard: --shard k --nshards n
python aggregate.py && python paper_numbers.py
```

The original runs used one NVIDIA DGX Spark (GB10), with PyTorch 2.12 and CUDA 13, running 3 shard processes in parallel; they took about one hour.

**Smoke test without data or GPU:**
```bash
XORLAB_FAKE=1 python grid.py --seeds 0 --no-lpips --out /tmp/smoke
```

## Notes

- **TF32 is disabled.** Threshold units are sensitive to rounding at u ≈ 0, so all runs use full fp32.
- **Seeds:** the seed fixes the initialisation of W, V and B and the data order. The Random test set is fixed across runs.
- **EMNIST** is used as torchvision delivers it (balanced split, transposed), as in `XORclassic/`.
- **LPIPS** is computed only in `A_*` runs. Two values are stored: `lpips_fixed` (all images, `normalize=True`; used in the paper) and `lpips_repo_style` (the first image of each 256-batch, un-normalised, as in `XORclassic/analysis.py`).
- **Readouts:** the multi-class perceptron rule (as in `XORclassic/`) and a least-squares (ridge) classifier with λ chosen on 10% of the training set. Both are computed on the trained encoder, on the initial encoder and, in `A_fc_ref`, on the raw pixels.
