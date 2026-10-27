# Control experiments (`review_controls/`)

Control experiments for the manuscript *"Toward defining loss functions in neuroscience: an XOR-based neuronal mechanism for local learning"* (v3: Tables 1–3, Fig. 6 and the control experiments of Sections 2.5 and 3).

`xorlab.py` re-implements the training rules of the main code in this repository (`XORclassic/`). With default settings it is bit-identical to it (initialisation, FC and CNN training, readouts), verified by `test_equivalence.py` on NVIDIA GB10 with PyTorch 2.12 / CUDA 13 (`test_equivalence_output.txt`). The rules are:
- same initialisation (same RNG call order: W, V, B) and learning rates;
- same signed-XOR masks and sign-DFA encoder rule;
- weights updated with the batch sum; biases with the mean over the batch (FC) or over batch and spatial positions (CNN);
- `max_corrections=3`;
- CNN with max-unpooling by indices; the encoder feedback `sign(conv(e, B))` is applied at every pre-pooling position, not only at the units selected by the pooling;
- the same perceptron readout.

On top of that it adds random seeds, logging and the switches needed for the controls. The CNN updates are vectorised with `einsum` instead of the original per-sample loop. This changes the order of the fp32 sums, but the summed terms are integer-valued (±1 × {0,1}), so the sums are exact and the results identical.

`test_equivalence.py` checks this against `XORclassic/` directly: initialisation, FC training and both readouts must be bit-identical; CNN forward passes must be bit-identical and single CNN updates must agree to fp32 rounding; the test also reports how closely short CNN training runs track each other (in the reference run: identical).

## Files

| File | Purpose |
|---|---|
| `xorlab.py` | Models, training rules, metrics and readouts |
| `grid.py` | Experiment grid (49 configurations × 5 seeds = 245 runs; the first 37 are those of the original grid). Resumable and shardable |
| `aggregate.py` | `results/summary.md`, `summary.csv`, `tables.tex`, `figs/` |
| `paper_numbers.py` | Prints every number quoted in the manuscript, with its source configuration |
| `test_equivalence.py` | Checks `xorlab.py` against `XORclassic/`; output of the reference run in `test_equivalence_output.txt` |
| `ridge_recheck.py` | Recomputes the ridge readout on an extended λ grid (1e-2–1e6) for raw pixels and the `A_*` / fixed encoders |
| `check_env.py` | Checks the environment and downloads MNIST, EMNIST-balanced and the LPIPS weights once |
| `setup_env.sh`, `Dockerfile`, `launch_shards.sh` | Optional: venv, NGC container, multi-node launch |
| `results/<config>/seed<k>.json` | Raw per-run results used in the paper (weights not included) |

## Mapping to the manuscript

| Manuscript | Configurations |
|---|---|
| Table 1 | `A_fc_ref` |
| Table 2 | `A_cnn_ref` (max-unpooling), `F_cnn_nearest` (nearest-neighbour upsampling) |
| Table 3 | Fully connected: `A_fc_ref`, `C_fc_frozen_enc`, `G_fc_unsigned`, `E_fc_randomtrain`, `J_fc_ste_lr3e-3`, raw-pixel readout (stored in `A_fc_ref`). Convolutional: `A_cnn_ref`, `C_cnn_frozen_enc`, `K_cnn_negb_frozen`, `K_cnn_negb_trained`, `J_cnn_ste_lr1e-1`, `F_cnn_nearest`, `F_cnn_nearest_frozen`, `K_cnn_nearest_negb_frozen`, `K_cnn_nearest_negb_trained` |
| Fig. 6 (bottleneck sweep) | `D_fc_h*`, `D_fc_frozen_h*`, with `A_fc_ref` / `C_fc_frozen_enc` at H = 1024 |
| Section 2.5 (STE learning rate) | `J_fc_ste`, `J_cnn_ste`, `J_*_ste_lr*` (selection by final training loss) |
| Section 2.4 (ridge λ) | `ridge_recheck.py` |
| Section 3.1 text | `G_cnn_unsigned`, `H_fc_raw_feedback`, `H_fc_raw_feedback_norm`, `I_fc_no_bias`, `I_cnn_no_bias`, `B_fc_corr1`, `B_fc_corr1_ep15`, `B_cnn_corr1`, `B_cnn_corr1_ep15` |
| Sections 3.2–3.3 text | configurations of Tables 1–3 and Fig. 6 (paired differences included), `E_cnn_randomtrain` |
| Not quoted | `E_fc_randomtrain_h0256`, `J_cnn_ste_nearest` (untuned), `K_*_bias_only` (uninformative: biases swamp the fixed small weights) |

`paper_numbers.py` prints these numbers grouped by manuscript section.

## Reproduce

```bash
pip install torch torchvision lpips matplotlib numpy
export XORLAB_DATA=$PWD/data TORCH_HOME=$PWD/torch_home
python check_env.py                                   # must print READY
python test_equivalence.py                            # must print PASS
python grid.py --seeds 0-4                            # or shard: --shard k --nshards n
python aggregate.py && python paper_numbers.py
```

The original runs used one NVIDIA DGX Spark (GB10), with PyTorch 2.12 and CUDA 13, running 3 shard processes in parallel; they took about one hour. The PyTorch version and device of every run are stored in its result file and listed in `results/summary.md` (section *Environment*).

**Smoke test without data or GPU:**
```bash
XORLAB_FAKE=1 python grid.py --seeds 0 --no-lpips --out /tmp/smoke
XORLAB_FAKE=1 python test_equivalence.py
```

## Notes

- **TF32 is disabled.** Threshold units are sensitive to rounding at u ≈ 0, so all runs use full fp32.
- **Seeds:** the seed fixes the initialisation of W, V and B and the data order. The Random test set is fixed across runs. Seeds are paired across configurations with the same shapes and training data (for example `A_*`, `C_*`, `F_*`, `K_*` of one architecture): the same seed gives the same W, V, B and the same batch order, so `aggregate.py` also reports seed-by-seed differences.
- **Unsigned control (`G_*`):** `|x − y|` carries no direction, so the update can only push outputs up and the network necessarily collapses to all-ones. It shows that the sign is logically required; the collapse is expected, not an empirical surprise.
- **Graded feedback (`H_*`):** with `B ~ N(0, 1/D)`, `|e·B| ≪ 1`, so `H_fc_raw_feedback` also lowers the encoder's effective learning rate. `H_fc_raw_feedback_norm` rescales `e·B` per update to mean |·| = 1, the step size of `sign(e·B)`, and isolates the effect of the sign itself.
- **STE baseline (`J_*`):** Adam, binary cross-entropy, 5 epochs (same data budget as the XOR rule). `J_*_ste` uses lr 1e-3; `J_*_ste_lr*` sweep 1e-4–1e-2 (CNN up to 3e-1). `paper_numbers.py` picks the learning rate by final training loss (never by test metrics).
- **EMNIST** is used as torchvision delivers it (balanced split, transposed), as in `XORclassic/`.
- **LPIPS** is computed only in `A_*` runs. Two values are stored: `lpips_fixed` (all images, `normalize=True`; used in the paper) and `lpips_repo_style` (the first image of each 256-batch, un-normalised, as in `XORclassic/analysis.py`). On 28×28 inputs the deepest VGG layers see 1–3 pixels, so LPIPS mainly reflects the early layers.
- **Readouts:** the multi-class perceptron rule (as in `XORclassic/`) and a least-squares (ridge) classifier with λ chosen on 10% of the training set from {1e-2, …, 1e3}. Both are computed on the trained encoder, on the initial encoder and, in `A_fc_ref`, on the raw pixels. `results/summary.md` lists the chosen λ and flags runs where it falls on the edge of the grid.
