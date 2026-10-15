# ViT training on Frontier

Trains a Vision Transformer on ERA5 weather data using the code from the
[SC25 deep learning at scale tutorial](https://github.com/NERSC/sc25-dl-tutorial),
in data-parallel (`dp`) mode.

Clone the training code at the pinned commit:
* `git clone https://github.com/saforem2/sc25-dl-tutorial.git`
* `cd sc25-dl-tutorial`
* `git checkout 7e91316d7af4af0f51bbfc7e9e7194418ae454ee`
* `cd ../`

| File | Purpose |
|---|---|
| `env.sh` | Paths for the code checkout, data, and logs. Export a variable to override it. |
| `config.yaml` | The `dp` training configs. |
| `submit.sh` | Slurm batch script; runs one config with Shifter. |

## Quick start

```bash
sbatch --reservation gm_thu submit.sh dp --local_batch_size=4
```

The default script will run on a single Frontier node in data-parallel mode, with average throughput 124 samples/sec. The total job time will be around 12 min for 2 epochs.
