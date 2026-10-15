# Vision Transformer (ViT) training on Aurora

Trains a Vision Transformer (ViT) on ERA5 weather data using the code from the
[SC25 deep learning at scale tutorial](https://github.com/NERSC/sc25-dl-tutorial),
data parallel across Aurora's Intel GPUs. 


| File | Purpose |
|---|---|
| `setup.sh` | Clones the training code at the pinned commit, applies the two patches below, and builds the venv. Run on a login node |
| `env.sh` | Paths for the Python environment, code checkout, data, and logs. |
| `ViT_aurora.yaml` | The training configurations |
| `submit_aurora.sh` | PBS batch script; sets up the environment and runs one or more configs. |
| `run_aurora.sh` | Launches one config of `train_mp.py` with [`ezpz`](https://github.com/saforem2/ezpz); called by `submit_aurora.sh`. |
| `sc25-persistent-workers.patch` | One-line DataLoader fix applied by `setup.sh` (see [Training code version](#training-code-version)). |
| `sc25-xpu-mp.patch` | Makes `train_mp.py` (model parallel) run on Intel XPU; applied by `setup.sh` (see [Model parallel](#model-parallel)). |

## Quick start

For the Genesis hackathon, there is a reservation in place. Use project
`GenesisHackathonOct26` and queue `R8914313` in your `qsub` commands.

```bash
./setup.sh
qsub -A GenesisHackathonOct26 -q R8914313 -l select=1 -v CONFIG=smoke submit_aurora.sh   # ~2 min check
qsub -A GenesisHackathonOct26 -q R8914313 -l select=4 submit_aurora.sh                   # dp, 4 node
```

Submit from this directory. PBS output goes to `vit-weather.o<jobid>` here.
Each run also writes `$LOGDIR/<config>-<timestamp>/run.log` (console output of all
ranks, plus the exact config used) and TensorBoard logs to
`$LOGDIR/<config>/<N>GPU/<timestamp>/`. `$LOGDIR` defaults to `logs/` in this directory.

## Configs

Each Aurora node has 6 GPUs with 2 tiles each; every tile is one data-parallel rank,
so a node runs 12 ranks. All configs use 16 samples per rank and the full
37,960-sample training set. Each config is sized for one node count: request that
many nodes with `-l select=N`.

| Config | Nodes | Ranks | Global batch | lr | Epochs (iterations) |
|---|---|---|---|---|---|
| `smoke` | 1 | 12 | 192 | 1.73e-3 | 2 × 10 steps on a subset  |
| `dp_1node` | 1 | 12 | 192 | 1.73e-3 | 4 (788) | 
| `dp_2node` | 2 | 24 | 384 | 2.45e-3 | 4 (392) | 
| `dp` | 4 | 48 | 768 | 3.46e-3 | 4 (196) | 
| `dp_8node` | 8 | 96 | 1536 | 4.90e-3 | 8 (192) | 
| `dp_16node` | 16 | 192 | 3072 | 6.93e-3 | 8 (96) |

<!--

| Config | Nodes | Ranks | Global batch | lr | Epochs (iterations) | Throughput |
|---|---|---|---|---|---|---|
| `smoke` | 1 | 12 | 192 | 1.73e-3 | 2 × 10 steps on a subset | — |
| `dp_1node` | 1 | 12 | 192 | 1.73e-3 | 4 (788) | ~175 samples/s |
| `dp_2node` | 2 | 24 | 384 | 2.45e-3 | 4 (392) | ~300 samples/s |
| `dp` | 4 | 48 | 768 | 3.46e-3 | 4 (196) | ~600–650 samples/s |
| `dp_8node` | 8 | 96 | 1536 | 4.90e-3 | 8 (192) | ~860 samples/s |
| `dp_16node` | 16 | 192 | 3072 | 6.93e-3 | 8 (96) | ~1,200 samples/s |


Throughput is the mean of epochs 2 onward, measured on 2026-10-13. Epoch-to-epoch
variation is large at 8 and 16 nodes (about ±20%).
-->

Compared with the Perlmutter `dp` config, these use the same model (ViT, embed dim
384, 12 layers), fp16 mixed precision, fused Adam, `torch.compile`, and 8 loader
workers. Two things differ:

- **Data loader:** PyTorch's `DataLoader` instead of DALI, which only runs on NVIDIA GPUs.
- **Batch size:** 256 isn't divisible across 12 ranks per node at 16 samples per rank,
  so the global batch is 192 per node, and the learning rate is scaled from the
  Perlmutter value: `lr = 2e-3 * sqrt(global_batch / 256)`.

Training runs whole epochs only: `num_iters` is rounded down to a multiple of the
epoch length, so a value below one epoch trains nothing. The first epoch includes
`torch.compile` (about 1–1.5 extra minutes) and warmup; use epochs 2 onward for
throughput. At 8 and 16 nodes an epoch is only 24 or 12 steps, so those configs
run 8 epochs.

## Model parallel

`run_aurora.sh` runs `train_mp.py`, which adds tensor (`tp`) and context (`cp`)
parallelism on top of data parallelism: `dp = ranks / (tp * cp)`. With
`tp = cp = 1` (the `smoke` and `dp_*` configs) it is plain data parallel. To run
the original `train.py` instead, set `TRAIN_SCRIPT=train.py`
(`qsub -v CONFIG=dp,TRAIN_SCRIPT=train.py ...`).

| Config | Nodes | tp × dp | Global batch | lr | Epochs (iterations) |
|---|---|---|---|---|---|
| `mp_smoke` | 1 | 4 × 3 | 48 | 8.66e-4 | 2 × 10 steps on a subset |
| `mp_smoke_2node` | 2 | 4 × 6 | 96 | 1.22e-3 | 2 × 10 steps on a subset |
| `mp_1node` | 1 | 4 × 3 | 48 | 8.66e-4 | 2 (1580) |
| `mp` | 4 | 4 × 12 | 192 | 1.73e-3 | 4 (788) |

These use the Perlmutter `mp` model (embed dim 1024), 16 samples per dp group,
`lr = 1e-3 * sqrt(global_batch / 64)`, and no `torch.compile`. They have not been
benchmarked on Aurora yet. `tp` must divide the 8 attention heads; ranks of a
`tp` group are consecutive, so `tp` up to 12 stays within a node. Change the split
with `--tensor_parallel` / `--context_parallel`, keeping `global_batch_size`
divisible by `dp`:

```bash
qsub -A GenesisHackathonOct26 -q R8914313 -l select=1 -v CONFIG=mp_smoke submit_aurora.sh
qsub -A GenesisHackathonOct26 -q R8914313 -l select=1 -v CONFIG=mp_1node,EXTRA_ARGS="--tensor_parallel=2" submit_aurora.sh
```

<!--
`sc25-xpu-mp.patch` ports `train_mp.py` the way upstream ported `train.py`:
device helpers instead of `torch.cuda.*`, device memory reporting instead of
`pynvml`, `ezpz.setup_torch()` under `mpiexec`, the `xccl` backend in
`utils/comm.py` instead of hard-coded NCCL, and SUM-then-divide instead of
`ReduceOp.AVG`. The Transformer Engine backend (`mp_te*` on Perlmutter) is
NVIDIA-only.


## Variations


```bash
# Several configs in one allocation, run in turn
qsub -A <project> -l select=1 -v CONFIG="smoke dp_1node" submit_aurora.sh

# Extra train.py arguments
qsub -A <project> -v CONFIG=dp,EXTRA_ARGS="--num_iters=49" submit_aurora.sh

# Write logs somewhere else (e.g. your project space)
qsub -A <project> -v CONFIG=dp,LOGDIR=/lus/flare/projects/<project>/$USER/vit-logs submit_aurora.sh
```

`submit_aurora.sh` defaults to the `debug-scaling` queue (1–256 nodes, 1 hour max,
one running job per user). Use `-q debug` for 1–2 nodes, `-q capacity` for up to
16 nodes.


To run interactively:

```bash
qsub -I -A GenesisHackathonOct26 -q R8914313 -l select=1 -l walltime=1:00:00 
# on the compute node, from this directory:
source env.sh
module use /soft/modulefiles; module load $FRAMEWORKS_MODULE
source $VENV/bin/activate
bash run_aurora.sh smoke
```

## Environment

`setup.sh` builds a small venv on top of the `frameworks/2026.1.0` module, which
provides PyTorch with Intel GPU (XPU) support, h5py, and mpi4py. The venv adds
`ezpz` (pinned commit, from GitHub; the PyPI package of that name is unrelated),
`ruamel.yaml`, and `tensorboard`. Run `setup.sh` on a login node, since compute
nodes don't have direct internet access.

The data is at `/lus/flare/projects/datasets/sc25_tutorial_data` (readable by all
Aurora users), so jobs request the `flare` filesystem.

`run_aurora.sh` sets `TMPDIR=/tmp`, because PBS's per-job `TMPDIR` is long enough
that DataLoader workers fail with `OSError: AF_UNIX path too long`.

## Training code version

`setup.sh` checks out commit `7e91316` of the `sunspot-xpu-support` branch of
[saforem2/sc25-dl-tutorial](https://github.com/saforem2/sc25-dl-tutorial)
(upstream PR #4). That branch makes `train.py` run on NVIDIA, AMD, and Intel GPUs,
so every system in these examples uses the same commit.

On top of it, `setup.sh` applies `sc25-persistent-workers.patch`, which keeps the
training DataLoader workers alive between epochs. Without it, PyTorch starts new
workers at every epoch, which costs about 25 s per epoch on Aurora. That cost
dominates the short epochs at 4–16 nodes: throughput drops to about 460 samples/s
at 4 nodes and 760 at 16. Each job prints the commit it ran in `run.log`.
