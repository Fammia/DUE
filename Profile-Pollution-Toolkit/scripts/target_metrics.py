"""Target-item hit rate among real users who never consumed the item."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Mapping, Sequence


def target_hit_rates(
    *,
    user_ids: Sequence[int],
    topk_lists: Sequence[Sequence[int]],
    targets: Sequence[int],
    consumed: Mapping[int, Iterable[int]],
    ks: Sequence[int] = (10, 20, 50),
    n_real_users: int = 6040,
) -> Dict[int, dict]:
    """
    consumed: original train∪valid∪test items per user, plus any injected
    train/valid items that should exclude the user (做法二 hosts).
    Users >= n_real_users are ignored.
    """
    consumed_sets = {int(u): set(int(x) for x in items) for u, items in consumed.items()}
    kmax = max(ks)
    out: Dict[int, dict] = {}
    for t in targets:
        t = int(t)
        hits = {k: 0 for k in ks}
        n_elig = 0
        for u, recs in zip(user_ids, topk_lists):
            u = int(u)
            if u < 0 or u >= n_real_users:
                continue
            if t in consumed_sets.get(u, set()):
                continue
            n_elig += 1
            rec_list = list(recs)[:kmax]
            for k in ks:
                if t in rec_list[:k]:
                    hits[k] += 1
        row = {"n_eligible": n_elig}
        for k in ks:
            row[f"hr@{k}"] = (hits[k] / n_elig) if n_elig else 0.0
            row[f"hits@{k}"] = hits[k]
        out[t] = row
    return out


def write_metrics(
    path: str,
    *,
    recall: Sequence[float],
    ndcg: Sequence[float],
    overall_ks: Sequence[int],
    target_rows: Mapping[int, dict],
    extra: dict | None = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "recall": {f"@{k}": float(recall[i]) for i, k in enumerate(overall_ks)},
        "ndcg": {f"@{k}": float(ndcg[i]) for i, k in enumerate(overall_ks)},
        "targets": {str(t): v for t, v in target_rows.items()},
    }
    if extra:
        payload.update(extra)
    hrs50 = [v.get("hr@50", 0.0) for v in target_rows.values()]
    payload["thr@50_mean"] = float(sum(hrs50) / len(hrs50)) if hrs50 else 0.0
    payload["thr@50_all_zero"] = all(h == 0.0 for h in hrs50) if hrs50 else True
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def print_metrics(recall, ndcg, overall_ks, target_rows) -> None:
    print("################### TEST ######################")
    # Recall@50 and Recall@100 are the first two overall ranking cutoffs.
    r50 = recall[list(overall_ks).index(50)] if 50 in overall_ks else recall[0]
    r100 = recall[list(overall_ks).index(100)] if 100 in overall_ks else recall[-1]
    n50 = ndcg[list(overall_ks).index(50)] if 50 in overall_ks else ndcg[0]
    n100 = ndcg[list(overall_ks).index(100)] if 100 in overall_ks else ndcg[-1]
    print("Recall {:.4f}-{:.4f}".format(r50, r100))
    print("NDCG {:.4f}-{:.4f}".format(n50, n100))
    print("################### TARGET HR ######################")
    for t in sorted(target_rows):
        row = target_rows[t]
        print(
            "TARGET_HR item={} n_eligible={} HR@10={:.6f} HR@20={:.6f} HR@50={:.6f}".format(
                t,
                row["n_eligible"],
                row["hr@10"],
                row["hr@20"],
                row["hr@50"],
            )
        )
    hrs = [target_rows[t]["hr@50"] for t in target_rows]
    print(
        "TARGET_HR_SUMMARY mean@50={:.6f} all_zero_at_50={}".format(
            sum(hrs) / len(hrs) if hrs else 0.0,
            all(h == 0.0 for h in hrs),
        )
    )
