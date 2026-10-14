# Profile-Pollution-Toolkit

A toolkit for constructing controlled profile-poisoning scenarios and evaluating recommender robustness. It implements random, bandwagon, and segment profiles, with fake-user injection and existing-user profile pollution. GMF and LightGCN runners compare unweighted Base with DUE through recommendation quality, target exposure, and training-weight trajectories.

Profile construction uses the processed MovieLens-1M splits and `movies.dat`; see [data/README.md](../data/README.md). It changes training interactions while keeping validation/test splits unchanged. Run commands from the repository root.

Target item IDs are stored in `meta/targets.json`. To select new targets, run `python Profile-Pollution-Toolkit/scripts/select_targets.py` before generating patches; this replaces the preset IDs. Selection uses training popularity and genre. After selecting new targets, run an unpolluted evaluation to check initial target exposure.

```bash
python Profile-Pollution-Toolkit/scripts/generate_patches.py
python Profile-Pollution-Toolkit/scripts/materialize_dataset.py
python scripts/run_profile_pollution.py --gate
python scripts/run_profile_pollution.py --gate --execute --gpu-id 0
python Profile-Pollution-Toolkit/scripts/check_gate.py
```

`generate_patches.py` creates 60 lists per target for each builder—random, bandwagon, and segment—and assigns host users for host pollution. The six doses `n10` through `n60` are prefixes of each 60-list set. `materialize_dataset.py` creates 36 training variants (three builders × two injection modes × six doses) under `Profile-Pollution-Toolkit/data/`, with unchanged validation/test files from `data/movielens/`.

`--gate` evaluates one bandwagon/host-pollution/DUE/GMF/n30 configuration. `check_gate.py` checks whether it produces nonzero target exposure. Select a configuration with `--model`, `--stem`, and `--setting`; add `--execute` to run it. `--all --execute` runs the 148-configuration Base/DUE grid. Checkpoints are selected by real-user validation Recall@50. Evaluation records real-user test Recall/NDCG and target Hits@10/20/50; fake users are not scored. `Vanilla` selects unweighted Base, and `Full` selects DUE.

The model scripts are `NCF/main_profile_pollution.py` (GMF) and `LightGCN/main_profile_pollution.py`. Toolkit helpers are in `scripts/`: `config.py` defines the grid and output paths, `weight_ops.py` computes DUE weights, and `curve_recorder.py` records per-epoch summaries.

Generated profiles are saved under `Profile-Pollution-Toolkit/patches/`; evaluation metrics and weight curves are saved under `Profile-Pollution-Toolkit/log/`. Reference aggregates are in `results/pollution_overall_ranking.json` and `results/pollution_target_hits.json`.
