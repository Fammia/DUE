"""Profile-pollution configuration: job names, paths, and evaluation commands.

37 data settings (36 attack stems + no_attack) x GMF/LightGCN x Vanilla/Full
= 148 one-pot trainings.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

Model = str
Setting = str  # Vanilla | Full
Builder = str  # random | bandwagon | segment
Inject = str  # fakeuser | pollute
Budget = str  # n10..n60
Stem = str

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOOLKIT_ROOT = os.path.join(PROJECT_ROOT, "Profile-Pollution-Toolkit")
ORIG_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "movielens")

SEED = 123456789
N_REAL_USERS = 6040
FAKE_USER_START = 6040
N_TARGETS = 10
PROFILES_PER_TARGET = 60
SELECTED_K = 10
MIN_PROFILE_LEN = 2
DATASET_NAME = "movielens"
GPU_SORT = "mem"

MODELS: Sequence[Model] = ("GMF", "LightGCN")
SETTINGS: Sequence[Setting] = ("Vanilla", "Full")
BUILDERS: Sequence[Builder] = ("random", "bandwagon", "segment")
INJECTS: Sequence[Inject] = ("fakeuser", "pollute")
BUDGETS: Sequence[Budget] = ("n10", "n20", "n30", "n40", "n50", "n60")

# Clean-data, validation-selected MovieLens Recall@50 settings.
DEFAULT_FACTORS: Dict[Model, Tuple[float, float]] = {
    "GMF": (0.3, 0.25),
    "LightGCN": (0.1, 0.95),
}

NCF_EVAL_INTERVAL = 5
NCF_PATIENCE = 6
LGCN_EVAL_INTERVAL = 1
LGCN_PATIENCE = 10
EPOCH_EVAL = 10

SHUFFLE_SEED = 20260820

CURVE_COLUMNS: Sequence[str] = (
    "epoch",
    "n_clean",
    "n_noise",
    "n_filler",
    "n_selected",
    "n_target",
    "loss_clean",
    "loss_noise",
    "loss_filler",
    "loss_selected",
    "loss_target",
    "w_final_clean",
    "w_final_noise",
    "w_final_filler",
    "w_final_selected",
    "w_final_target",
    "w_base_clean",
    "w_base_noise",
    "w_base_filler",
    "w_base_selected",
    "w_base_target",
    "w_user_clean",
    "w_user_noise",
    "w_user_filler",
    "w_user_selected",
    "w_user_target",
    "w_item_clean",
    "w_item_noise",
    "w_item_filler",
    "w_item_selected",
    "w_item_target",
)

TARGET_ITEM_COLUMNS: Sequence[str] = (
    "epoch",
    "item_id",
    "w_item",
    "w_final_target_mean",
    "n_target_edges",
)

ROLE_IDS = {
    "noise": 0,
    "clean": 1,
    "filler": 2,
    "selected": 3,
    "target": 4,
}
ROLE_NAMES = ("clean", "noise", "filler", "selected", "target")


def budget_n(budget: Budget) -> int:
    if not str(budget).startswith("n"):
        raise ValueError(budget)
    return int(str(budget)[1:])


def optima(model: Model) -> Tuple[float, float]:
    return DEFAULT_FACTORS[model]


def data_stem(builder: Builder, inject: Inject, budget: Budget) -> Stem:
    return f"{builder}_{inject}_{budget}"


def extra_stems() -> List[Stem]:
    out: List[Stem] = []
    for builder in BUILDERS:
        for inject in INJECTS:
            for budget in BUDGETS:
                out.append(data_stem(builder, inject, budget))
    return out


def all_stems() -> List[Stem]:
    return ["no_attack"] + extra_stems()


def parse_stem(stem: Stem) -> Optional[Tuple[Builder, Inject, Budget]]:
    if stem == "no_attack":
        return None
    parts = stem.split("_")
    budget = parts[-1]
    inject = parts[-2]
    builder = "_".join(parts[:-2])
    return builder, inject, budget


def data_dir(stem: Stem) -> str:
    if stem == "no_attack":
        return ORIG_DATA_DIR
    return os.path.join(TOOLKIT_ROOT, "data", stem)


def meta_dir() -> str:
    return os.path.join(TOOLKIT_ROOT, "meta")


def targets_path() -> str:
    return os.path.join(meta_dir(), "targets.json")


def patches_dir() -> str:
    return os.path.join(TOOLKIT_ROOT, "patches")


def profiles_path(builder: Builder) -> str:
    return os.path.join(patches_dir(), f"{builder}_profiles_n60.json")


def hosts_path(builder: Builder) -> str:
    return os.path.join(patches_dir(), f"{builder}_hosts_n60.json")


def fake_roles_path(stem: Stem) -> str:
    return os.path.join(data_dir(stem), "fake_roles.csv")


def dump_dir(model: Model, stem: Stem, setting: Setting) -> str:
    return os.path.join(TOOLKIT_ROOT, "log", "dump", model, stem, setting)


def curves_path(model: Model, stem: Stem, setting: Setting) -> str:
    return os.path.join(dump_dir(model, stem, setting), "curves.csv")


def target_item_weights_path(model: Model, stem: Stem, setting: Setting) -> str:
    return os.path.join(dump_dir(model, stem, setting), "target_item_weights.csv")


def scores_path(model: Model, stem: Stem, setting: Setting) -> str:
    return os.path.join(TOOLKIT_ROOT, "log", "scores", model, stem, setting, "metrics.json")


def fake_uid(target_slot: int, profile_index: int) -> int:
    return FAKE_USER_START + target_slot * PROFILES_PER_TARGET + profile_index


def _csv_has_header_and_rows(path: str, min_rows: int = 1) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        return len(lines) >= 1 + min_rows
    except OSError:
        return False


def onepot_done(model: Model, stem: Stem, setting: Setting) -> bool:
    mpath = scores_path(model, stem, setting)
    if not os.path.isfile(mpath):
        return False
    try:
        with open(mpath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("best_epoch") is None:
            return False
        targets = payload.get("targets") or {}
        if len(targets) < N_TARGETS:
            return False
        for row in targets.values():
            if "hits@50" not in row:
                return False
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    if not _csv_has_header_and_rows(curves_path(model, stem, setting), min_rows=1):
        return False
    if not _csv_has_header_and_rows(target_item_weights_path(model, stem, setting), min_rows=N_TARGETS):
        return False
    return True


# Back-compat aliases used by leftover dump scripts
def dump_done(model: Model, stem: Stem, setting: Setting) -> bool:
    return onepot_done(model, stem, setting)


def score_done(model: Model, stem: Stem, setting: Setting) -> bool:
    return onepot_done(model, stem, setting)


def job_stem(model: Model, data: Stem, setting: Setting) -> str:
    return f"{model}__{data}__{setting}"


def iter_jobs() -> List[Tuple[Model, Stem, Setting, float, float]]:
    jobs: List[Tuple[Model, Stem, Setting, float, float]] = []
    for model in MODELS:
        fr, fu = optima(model)
        for stem in all_stems():
            for setting in SETTINGS:
                jobs.append((model, stem, setting, fr, fu))
    return jobs


def job_count() -> int:
    return len(iter_jobs())


def ncf_score_flags(model: Model) -> str:
    fr, fu = optima(model)
    return (
        f"--floor_ratio {fr} --factor_upper {fu} "
        f"--eval_interval {NCF_EVAL_INTERVAL} --patience {NCF_PATIENCE} "
        f"--epoch_eval {EPOCH_EVAL} --seed {SEED} --gpu_sort {GPU_SORT}"
    )


def lgcn_score_flags() -> str:
    fr, fu = optima("LightGCN")
    return (
        f"--floor_ratio {fr} --factor_upper {fu} "
        f"--eval_interval {LGCN_EVAL_INTERVAL} --patience {LGCN_PATIENCE} "
        f"--epoch_eval {EPOCH_EVAL} --seed {SEED} --gpu_sort {GPU_SORT}"
    )


def onepot_command(model: Model, stem: Stem, setting: Setting) -> str:
    dpath = data_dir(stem)
    mout = scores_path(model, stem, setting)
    extra = (
        f"--data_path {dpath} --setting {setting} --metrics_out {mout} "
        f"--n_real_users {N_REAL_USERS} --stem {stem}"
    )
    if model == "GMF":
        return f"python main_profile_pollution.py --model {model} {extra} {ncf_score_flags(model)}"
    return f"python main_profile_pollution.py {extra} {lgcn_score_flags()}"


def score_command(model: Model, stem: Stem, setting: Setting) -> str:
    return onepot_command(model, stem, setting)


def dump_command(model: Model, stem: Stem, setting: Setting) -> str:
    return onepot_command(model, stem, setting)


def gate_job() -> Tuple[Model, Stem, Setting]:
    """Bandwagon host-profile pollution, GMF, full DUE, dose n30."""
    return "GMF", data_stem("bandwagon", "pollute", "n30"), "Full"


def stem_priority(stem: Stem) -> int:
    if stem == "no_attack":
        return 0
    if stem.startswith("bandwagon_pollute_"):
        return 1
    return 2


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--gate":
        gm, gs, gset = gate_job()
        print(score_command(gm, gs, gset))
        raise SystemExit(0)
    jobs = iter_jobs()
    print(f"Profile-pollution jobs: {len(jobs)}")
    print(f"extra datasets: {len(extra_stems())}")
    gm, gs, gset = gate_job()
    print(f"gate: {score_command(gm, gs, gset)}")
    assert len(extra_stems()) == 36, len(extra_stems())
    assert len(jobs) == 148, len(jobs)
    assert len(all_stems()) == 37
