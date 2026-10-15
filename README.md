# Genesis hackathon I5: ViT training examples

Examples for the Genesis hackathon session **I5: Getting started with pretraining
foundation models on DOE supercomputers**.

Here we train a Vision Transformer (ViT) on ERA5 weather data on three DOE
supercomputers, using the code from the
[SC25 deep learning at scale tutorial](https://github.com/NERSC/sc25-dl-tutorial).

Each system has its own folder with setup, configs, a batch script, and a README.
Start with the README for the system you're using.

| System | Folder  | Notes |
|---|---|---|
| Perlmutter (NERSC) | [`perlmutter`](perlmutter)  | Slurm + Shifter |
| Aurora (ALCF) | [`aurora`](aurora)  | PBS + [ezpz](https://github.com/saforem2/ezpz);|
| Frontier (OLCF) | [`frontier`](frontier) | Slurm |

## Hackathon reservations

Use the following hackathon allocations. The Quick start in each system's README
shows where they go.

| System | Account / project | Reservation / queue | Available |
|---|---|---|---|
| Perlmutter | `m5429_g` | `gm_hackathon26_session5` (250 nodes) | During the session: 2026-10-15, 13:30–15:00 Pacific |
| Aurora | `GenesisHackathonOct26` | `R8914313` | All hackathon week |

After the session, Perlmutter also has a 100-node reservation on the same account:
`gm_hackathon26_day3` (2026-10-16, 07:00–13:30 Pacific).

## Training code

All three systems use commit `7e91316` of the `sunspot-xpu-support` branch of
[saforem2/sc25-dl-tutorial](https://github.com/saforem2/sc25-dl-tutorial), which makes
the tutorial's `train.py` run on NVIDIA, AMD, and Intel GPUs. The setup steps clone it
for you. On Aurora, `setup.sh` also applies two patches on top of it: one keeps
DataLoader workers alive between epochs, and one ports the model-parallel
`train_mp.py` to Intel GPUs.

## Credits

- Perlmutter example: Steve Farrell
- Aurora example: Kyle Felker, Murali Emani
- Frontier example: Aristeidis Tsaris
- Training code: the [SC25 deep learning at scale tutorial](https://github.com/NERSC/sc25-dl-tutorial)
  by NERSC, with device portability by Sam Foreman
