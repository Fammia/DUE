# DUE: Defer Until Evidence

DUE combines loss-rank interaction weighting with support-gated asymmetric user and item reweighting for implicit-feedback recommendation. Implementations support GMF, NeuMF, CDAE, and LightGCN. The profile-pollution toolkit provides MovieLens-1M robustness evaluation for GMF and LightGCN.

## Contents

| Path | Purpose |
| --- | --- |
| `NCF/` | GMF and NeuMF models, training, and evaluation |
| `CDAE/` | CDAE model, training, and evaluation |
| `LightGCN/` | LightGCN model, training, and evaluation |
| `Profile-Pollution-Toolkit/` | Profile construction, target-exposure evaluation, and weight recording |
| `common/` | Automatic GPU selection |
| `scripts/` | Input checks and training launchers |
| `data/` | Input layout and split checksums |
| `results/` | Reference ranking and profile-pollution metrics |
| `tests/` | Weighting and utility tests |

## Environment

Training requires Python, PyTorch, and an NVIDIA GPU. The tested environment uses Python 3.11 and PyTorch 2.1.2 with CUDA 12.1. Install a CUDA-compatible PyTorch build, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data

Download the processed splits from [UDT](https://github.com/hyNTU/UDT/tree/cc2c8706105ee15b92a110bc07eb31e75f64d8d9/data) into `data/<dataset>/`. Supported dataset names are `movielens`, `yelp`, and `amazon_book`. See [data/README.md](data/README.md) for file names, formats, and the additional `movies.dat` input used for profile construction.

## Training and Evaluation

Run these commands from the repository root:

```bash
python scripts/check_data.py --dataset movielens --paper-hash
python scripts/run_main.py --dataset movielens --backbone GMF
python scripts/run_main.py --dataset movielens --backbone GMF --execute --gpu-id 0
```

`check_data.py --paper-hash` verifies the split checksums. `run_main.py` prints the training command by default; `--execute` runs it. The `--backbone` choices are `GMF`, `NeuMF`, `CDAE`, and `LightGCN`; `--backbone all --dataset all --execute` runs all twelve configurations sequentially. The launcher uses the configured `(floor_ratio, factor_upper)` pairs and seed `123456789`; change the seed with `--seed` and set a GPU with `--gpu-id`.

Checkpoints are selected by validation Recall@50. Test evaluation reports Recall@50/100 and NDCG@50/100. Logs are written to `outputs/main/`, and checkpoints to the selected backbone's `models/` directory. Reference scores are in [results/main_due.csv](results/main_due.csv).

## Profile-Pollution-Toolkit

The toolkit constructs random, bandwagon, and segment profiles, using either fake-user injection or pollution of existing user profiles. GMF and LightGCN evaluation compares unweighted Base with DUE using ranking quality and target exposure. Target IDs are stored in `Profile-Pollution-Toolkit/meta/targets.json`.

```bash
python scripts/check_data.py --dataset movielens --paper-hash --profile-pollution
python Profile-Pollution-Toolkit/scripts/generate_patches.py
python Profile-Pollution-Toolkit/scripts/materialize_dataset.py
python scripts/run_profile_pollution.py --model GMF --stem no_attack --setting Full
python scripts/run_profile_pollution.py --model GMF --stem no_attack --setting Full --execute --gpu-id 0
```

Construction writes profiles to `Profile-Pollution-Toolkit/patches/` and training variants to `Profile-Pollution-Toolkit/data/`. The evaluation launcher prints commands by default; `--execute` runs them, and `--all --execute` runs the 148-configuration grid. Metrics and optional weight curves are saved under `Profile-Pollution-Toolkit/log/`. See [Profile-Pollution-Toolkit/README.md](Profile-Pollution-Toolkit/README.md) for configuration details.

## Tests

```bash
python -m unittest discover -s tests -v
```
