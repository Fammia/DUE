# Reference Results

`main_due.csv` contains DUE test Recall@50/100 and NDCG@50/100 for each backbone and dataset, using seed `123456789` and four-decimal scores.

`pollution_overall_ranking.json` contains overall ranking metrics for the profile-pollution configurations. `pollution_target_hits.json` contains target-hit summaries for the corresponding injection doses.

Training writes main-run scores to `outputs/main/` logs and profile-pollution metrics to `Profile-Pollution-Toolkit/log/scores/.../metrics.json`.
