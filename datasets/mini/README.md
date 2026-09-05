---
configs:
- config_name: default
  data_files:
  - path: data/mini-*.parquet
    split: mini
- config_name: full
  data_files:
  - path: data/full-*.parquet
    split: full
homepage: https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026
language:
- code
license: cc-by-4.0
pretty_name: 'WhestBench 2026: ARC White-Box Estimation Challenge'
repository: https://github.com/AIcrowd/whestbench
size_categories:
- 1K<n<10K
tags:
- whestbench
- alignment
- neural-network-statistics
- benchmark
- white-box
- multi-split
task_categories:
- other
---

<p align="center">
  <a href="https://github.com/AIcrowd/whestbench">
    <img src="https://raw.githubusercontent.com/AIcrowd/whestbench/main/assets/logo/logo.png" width="320" alt="WhestBench logo">
  </a>
</p>

<p align="center">
  Organized by:
  <a href="https://www.alignment.org/"><b>Alignment Research Center (ARC)</b></a>,
  <a href="https://www.aicrowd.com/"><b>AIcrowd</b></a>
</p>

# WhestBench 2026: ARC White-Box Estimation Challenge

<p align="center">
  <a href="https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026"><img alt="Challenge" src="https://img.shields.io/badge/AIcrowd-Challenge_Page-f0524d?style=for-the-badge"></a>
  <a href="https://github.com/AIcrowd/whestbench"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-AIcrowd%2Fwhestbench-181717?style=for-the-badge&logo=github&logoColor=white"></a>
  <a href="https://github.com/AIcrowd/whest-starterkit"><img alt="Starter Kit" src="https://img.shields.io/badge/Starter_Kit-whest--starterkit-f57c00?style=for-the-badge&logo=github&logoColor=white"></a>
  <a href="https://aicrowd.github.io/whestbench-explorer/"><img alt="MLP Explorer" src="https://img.shields.io/badge/MLP_Explorer-Interactive-7e57c2?style=for-the-badge"></a>
  <a href="https://github.com/AIcrowd/flopscope"><img alt="flopscope" src="https://img.shields.io/badge/FLOP_Tracking-flopscope-009688?style=for-the-badge&logo=github&logoColor=white"></a>
  <a href="https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026/tree/v2-phase2"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97-View_on_HF_Hub-ffd54f?style=for-the-badge"></a>
</p>

WhestBench is a benchmark for *white-box activation estimation*: given the weights of a randomly initialized ReLU multi-layer perceptron (MLP) and a strict floating-point-operation (FLOP) budget, predict the average post-activation value of every neuron when the network is fed standard Gaussian inputs.

**This is the WhestBench 2026 Public Dataset Release** — pre-baked MLPs paired with their ground-truth activation statistics. Two **independent** splits, baked with **disjoint seeds**:

- **`mini`** (100 MLPs) — the development split. Small enough to download in seconds. Iterate on your estimator here.
- **`full`** (1,000 MLPs) — the canonical evaluation surface. Run against this once your estimator is dialed in.

`mini` is **not** a subset of `full`. They share no MLPs — verify with the `mlp_seed` column.

## Quick start

The pure HuggingFace path (no whestbench install required):
```python
from datasets import load_dataset

# Develop against mini — 100 MLPs, downloads in seconds.
# `mini` is the default config of this repo, so no name= is required.
mini = load_dataset("aicrowd/arc-whestbench-public-2026", revision="v2-phase2", split="mini")
print(mini[0]["mlp_name"])

# Lock in your numbers against full — 1,000 MLPs.
# `full` is a separate config; pass the name explicitly. mini and full are
# independent — loading full does NOT also pull mini (or vice versa).
full = load_dataset("aicrowd/arc-whestbench-public-2026", "full", revision="v2-phase2", split="full")
print(full[0]["mlp_name"])
```

The whestbench convenience wrapper (adds schema validation + `metadata.json` access):

```python
import whestbench

ds = whestbench.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v2-phase2", split="mini")
for mlp in whestbench.iter_mlps(ds):
    # `mlp` is a whestbench.MLP with .weights, .seed, .name, .width, .depth
    ...

provenance = whestbench.metadata(ds)
print(provenance["n_samples"], provenance["created_at_utc"])
```

Run an estimator end-to-end via the CLI:
```bash
whest run \
    --estimator my_estimator.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2
# (automatically uses metadata.default_split: "mini"; pass --split <name> to override)
```

➡️ **New to the challenge?** Head over to the **[WhestBench starter kit](https://github.com/AIcrowd/whest-starterkit)** for a worked example estimator, the recommended project layout, FLOP-tracking patterns with [`flopscope`](https://github.com/AIcrowd/flopscope), local testing tips, and the submission workflow.


## Schema

Each row is one MLP. Eight columns:

| Column | Type / shape | What this is |
|---|---|---|
| `mlp_id` | `int32` | 0-based index of this MLP within the dataset (the absolute index across all parallel-bake slices). |
| `mlp_name` | `string` | Stable, deterministic human-readable slug like `"danielle-johnson"`, derived from `mlp_seed`. Useful for log lines; carries no information beyond `mlp_seed`. |
| `mlp_seed` | `int64` | Per-MLP seed exposed in the dataset. Under seed_protocol 3.0 (`whestbench_explicit_per_mlp_seeds`), this is the **input** seed for the MLP — `MLP.seed` (the estimator seed) is derived locally. Under seed_protocol 2.0 (`whestbench_seedsequence_hierarchy`, legacy), this is the **derived** estimator seed itself. Estimators read `mlp.seed` and see the same kind of value in both protocols (a deterministic int derived from the dataset's seed material). |
| `weights` | `float32[depth, width, width]` | The MLP's layer weight matrices. The network has **no biases** and **no separate linear output layer** — every weight matrix is followed by a ReLU. Layer `l` computes `h_l(x) = max(0, h_{l-1}(x) @ W_l)` with `h_0(x) = x`, i.e. the activation is a **row vector multiplied on the left** of the weight matrix (equivalently `max(0, W_l^T @ h_{l-1}(x))` in column-vector form). Weights are drawn i.i.d. from `N(0, 2/width)` (He initialization) at bake time. |
| `all_layer_means` | `float32[depth, width]` | **Ground truth.** Entry `[l, j]` is the empirical mean of neuron `j`'s post-ReLU output at layer `l`, averaged over **N = 1,000,000,000** independent Gaussian inputs: `E_{x ~ N(0, I)}[ h_l(x)_j ] ≈ (1/N) Σ_i h_l(x_i)_j`. Computed by direct Monte Carlo. **This is what your estimator predicts.** |
| `final_means` | `float32[width]` | The last row of `all_layer_means` — i.e. `E[h_{depth}(x)_j]` for each output neuron `j`, again over N = 1,000,000,000 samples. Materialised as its own column because the **primary scoring metric** (`final_layer_mse`) only looks at this row. |
| `avg_variance` | `float64` | Per-MLP mean of the per-neuron output variance at the final layer: `(1/width) Σ_j Var[h_{depth}(x)_j]`. A single scalar per MLP, computed alongside the means over the same Monte Carlo draws. Shipped as **diagnostic provenance** — useful for normalising your own MSE locally or as input to variance-aware estimators. **Not** consumed by the active scoring formula (the score is `mse_final · max(0.1, C_m / B_m)`). |
| `sampling_budget_breakdown` | `string` (JSON) | FLOP accounting for the bake that produced the ground truth for **this** row — useful as provenance. **Not** related to the estimator's FLOP budget at evaluation time: the reference bake is not budget-limited, so `flop_budget` is recorded equal to `flops_used` and `flops_remaining` is `0`. Decode with `json.loads(...)` to get a dict with keys: `flop_budget`, `flops_used`, `flops_remaining`, `wall_time_s`, `flopscope_backend_time_s`, `flopscope_overhead_time_s`, `residual_wall_time_s`, `time_source`, `by_namespace` (per-namespace nested breakdown: each namespace key maps to its own `{flops_used, calls, flopscope_backend_time_s, flopscope_overhead_time_s, operations}`, where each entry of `operations` is `{flop_cost, calls, flopscope_backend_time_s, flopscope_overhead_time_s}`), and `provenance` (how the counts were derived, including the `chunk_size` and `n_chunks` the bake actually used — the two values the reproduction recipe above otherwise cannot recover). |



## How the ground truth was made

> **Monte Carlo with N = 1,000,000,000 samples per MLP.** Every entry in `all_layer_means` and `final_means` is the empirical mean over this many independent standard-Gaussian input draws. The per-neuron standard deviation of the published value is `sqrt(Var[h_j]/N)`, so the ground-truth MSE is `avg_variance / N` — both computable exactly for this release from the published `avg_variance` column, and orders of magnitude smaller than any meaningful estimator gap.

**Input distribution.** Every Monte Carlo sample is a fresh `x ~ N(0, I)` of shape `(width,)`. The same input is forward-propagated through all `depth` layers in one pass, so the per-layer means at indices `[0..depth-1]` share the same input draws.

**Estimator.** Sums of post-ReLU activations are accumulated in `float64` for numerical stability, then divided by N at the end and downcast to `float32`. The final-layer variance scalar (`avg_variance`) comes from `E[h²] - (E[h])²` over the same N draws.

**Compute.** Sampling is chunked so memory stays bounded (`~4MB` per chunk on the flopscope CPU backend; tunable on the torch GPU backend via `chunk_size`). On the torch backend, under matched determinism config (`torch.use_deterministic_algorithms=True`, `cudnn.deterministic=True`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`) on a fixed `torch` version and GPU architecture, the bake is **bit-exact**: `weights`, `all_layer_means`, and `final_means` reproduce byte-for-byte; `avg_variance` agrees to `1e-12` relative tolerance — a deliberately loose bound on the cancellation in the `(sum_sq/n − mean²)` step, not a tight one (one float64 ULP is `2⁻⁵² ≈ 2.2×10⁻¹⁶`). The two backends (flopscope CPU and torch GPU) produce statistically equivalent output: being two independent Monte-Carlo estimates, their means differ by `sqrt(2·avg_variance/N)`, of order `10⁻⁵` at N=10⁹.

## Versions

Released per competition round. Every version shares the same splits, schema 3.0 and seed protocol 3.0; what changes is the MLP shape and the evaluation budget.

| Version | Round | Shape (width × depth) | MLPs | Forward-pass FLOPs / sample | Budget / MLP | Effective MC samples | Created |
|---|---|---|---|---|---|---|---|
| [**`v2-phase2`**](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026/tree/v2-phase2) | **Phase 2 ← this version** | **1024 × 16** | **1,100** | **33,554,432** | **2⁴¹ ≈ 2.20 × 10¹²** | **65,536** | **2026-08-13** |
| [`v1-phase1`](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026/tree/v1-phase1) | Phase 1 | 256 × 32 | 1,100 | 4,194,304 | ≈ 2.72 × 10¹¹ | ≈ 64,850 | 2026-06-16 |
| [`v1-warmup`](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026/tree/v1-warmup) | warm-up | 256 × 8 | 1,100 | 1,048,576 | 6.8 × 10¹⁰ | ≈ 64,850 | 2026-05-26 |

**How to read the last three columns.** Evaluating one Gaussian input through a `width × depth` MLP costs `2 × depth × width²` FLOPs — that is **Forward-pass FLOPs / sample**. It is exact, and it does not drift between flopscope releases: verified to the unit against flopscope 0.4.1 — the release that baked `v1-warmup` — as well as 0.11.0 and current `main`. The forward pass runs entirely in float32, and flopscope's dtype-aware pricing does not touch it.

Dividing **Budget / MLP** by it gives **Effective MC samples** — an *upper bound* on how many inputs a naive Monte-Carlo estimator can afford. A real estimator also pays for random-number generation, dtype casts, and accumulating a running mean; in the reference bake that overhead is another 0.4–1%. float64 work is metered at twice the float32 rate, so exactly how much an estimator pays depends on how it is written.

Each phase's budget is calibrated to buy roughly the same number of naive Monte-Carlo samples per MLP — about 65,000 — so what changes between rounds is the MLP shape, not how much brute force the budget affords. The error a naive estimator reaches at that sample count is the reference point a structured estimator is measured against.

For reference, the ground truth in every version is estimated from **10⁹** samples per MLP, roughly 15,000× more than the budget affords, so its own sampling noise sits well below the errors the leaderboard is expected to separate.

Newer versions may exist — see [all versions](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026/tags). This table lists what was published as of this release; earlier cards are never rewritten, so a card at an older tag lists fewer versions than this one.

```python
# Pin the version. The MLP shape differs between them, so an unpinned load can
# silently hand you a different network than the one your estimator expects.
load_dataset("aicrowd/arc-whestbench-public-2026", revision="v2-phase2", split="…")
```

## Dataset summary — `v2-phase2`

|  |  |
|---|---|
| Splits | mini, full |
| MLPs total | 1100 |
| `mini` split MLPs | 100 |
| `full` split MLPs | 1,000 |
| Width | 1024 |
| Depth | 16 |
| Monte Carlo samples per MLP (N) | **1,000,000,000** |
| Schema version | 3.0 |
| Seed protocol | `whestbench_explicit_per_mlp_seeds` v3.0 |

## Working with this dataset

Three ways in, fastest first:

- **Score an estimator** — `whest run --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2` runs your estimator end-to-end against the default `mini` split (add `--split <name>` to switch). No manual download step — the data is fetched and cached for you.
- **Load in Python** — `whestbench.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v2-phase2", split="mini")` returns a HuggingFace `Dataset` with schema validation plus `whestbench.metadata(ds)` provenance; iterate ready-to-use MLP objects with `whestbench.iter_mlps(ds)`.
- **Raw rows** — plain `datasets.load_dataset(...)` works too, if you'd rather not install whestbench (see [Quick start](#quick-start)).

📖 **Full walkthrough** — choosing a split, FLOP budgeting, the estimator contract, and the submission flow — see **[Use Evaluation Datasets](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/how-to/use-evaluation-datasets.md)** in the starter kit.

## Reproducibility

This dataset was baked with:

- **Backend:** `torch`
- **Seed protocol:** `whestbench_explicit_per_mlp_seeds` v3.0

- **Created (UTC):** `2026-08-13T15:11:58.709376+00:00`



This dataset was baked with **seed_protocol 3.0 (explicit per-MLP seeds)**: every MLP's input seed lives in the parquet `mlp_seed` column, so the same MLPs are reproducible from the published data alone. You almost never need to do this — but if you want to regenerate the dataset, expand the recipe below.

<details>
<summary><b>Re-bake this dataset locally from its seeds</b></summary>

**1. Extract the per-MLP seeds** into one JSON file per split:

```python
import json

import whestbench

for split in ["mini", "full"]:
    ds = whestbench.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v2-phase2", split=split)
    json.dump([int(r["mlp_seed"]) for r in ds], open(f"{split}-seeds.json", "w"))
```

**2. Re-bake** each split with the same seeds:

> The seeds fix the **weights** exactly. Reproducing the **means** bit-for-bit additionally
> requires matching the chunking the original bake used — `--chunk-size` and
> `--mlps-per-batch` — because float64 accumulation is not associative, so a different
> chunk decomposition sums the same samples in a different order.
>
> **`--chunk-size` is published per row**: decode `sampling_budget_breakdown` and read
> `provenance.chunk_size` (with `provenance.n_chunks` as a cross-check). Pass that value
> back and the accumulation order matches. `--mlps-per-batch` is *not* recorded anywhere;
> both flags auto-resolve from free VRAM when omitted. So with `--chunk-size` supplied the
> recipe below reproduces the same MLPs and the same accumulation order, and without it,
> the same MLPs and statistically equivalent means rather than identical bytes.

```bash
whest dataset bake \
    --n-mlps 100 \
    --n-samples 1000000000 \
    --width 1024 --depth 16 \
    --split mini \
    --mlp-seeds mini-seeds.json \
    --output ./mini-rebake \
    --torch --device cuda
whest dataset bake \
    --n-mlps 1000 \
    --n-samples 1000000000 \
    --width 1024 --depth 16 \
    --split full \
    --mlp-seeds full-seeds.json \
    --output ./full-rebake \
    --torch --device cuda

```

</details>


The pins required for the bit-exactness guarantee (see *Compute* above) are recorded per hardware configuration in `metadata.hardware_fingerprints[*].bake_config`, alongside the `whestbench`, `flopscope`, `torch` and `numpy` versions that configuration used.

## Provenance

Assembled from multiple partial bakes merged via `whest dataset merge` across a fleet of NVIDIA GeForce RTX 3090 GPUs, in **1** distinct hardware configuration.

**Configuration 1** — 1100 partials
- **GPU**: NVIDIA GeForce RTX 3090 (compute capability 8.6)
- **CPU**: x86_64 (72 logical cores)
- **RAM**: 270 GB
- **Python**: 3.11.10 · **PyTorch**: 2.4.1+cu124 (device=cuda) · **NumPy**: 2.4.6
- **whestbench**: 0.16.0 · **flopscope**: 0.12.0
- **Determinism**: `torch.use_deterministic_algorithms=True` · `cudnn.deterministic=True` · `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- **CUDA drivers observed**: 550.120, 550.163.01, 560.35.03, 565.77, 570.144, 570.158.01, 570.195.03, 570.211.01, 570.86.15, 580.126.09, 580.126.20, 580.142, 580.159.03, 580.173.02, 580.95.05, 590.48.01, 595.71.05, 595.84
- **Linux kernels observed**: 5.15.0-119-generic, 5.15.0-139-generic, 5.15.0-160-generic, 5.15.0-164-generic, 5.15.0-171-generic, 5.15.0-174-generic, 5.15.0-176-generic, 5.15.0-179-generic, 5.15.0-181-generic, 5.15.0-185-generic, 5.15.0-186-generic, 5.15.0-187-generic, 6.17.0-35-generic, 6.5.0-35-generic, 6.8.0-101-generic, 6.8.0-106-generic, 6.8.0-107-generic, 6.8.0-110-generic, 6.8.0-111-generic, 6.8.0-117-generic, 6.8.0-124-generic, 6.8.0-134-generic, 6.8.0-136-generic, 6.8.0-137-generic, 6.8.0-40-generic, 6.8.0-52-generic, 6.8.0-59-generic, 6.8.0-60-generic, 6.8.0-79-generic, 6.8.0-87-generic, 6.8.0-90-generic, 6.8.0-94-generic, 7.0.0-28-generic


## Citation

If you use this dataset, please cite the challenge:


```bibtex
@misc{whestbench2026,
  title        = {{WhestBench 2026: ARC White-Box Estimation Challenge}},
  author       = {{Alignment Research Center} and {AIcrowd}},
  year         = {2026},
  howpublished = {\url{https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026}},
}
```


## License

Released under **CC-BY-4.0**. Use is encouraged for research, competition entries, and educational material; please credit the WhestBench team and the AIcrowd challenge.