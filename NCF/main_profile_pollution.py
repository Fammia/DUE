"""GMF profile-pollution evaluation: ranking, target exposure, and weight traces."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import toolz

import data_utils
import evaluate
import model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from common.gpu_utils import get_gpu_id_str_sorted  # noqa: E402

_TOOLKIT = os.path.join(PROJECT_ROOT, "Profile-Pollution-Toolkit", "scripts")
sys.path.insert(0, _TOOLKIT)
from five_way_curves import aggregate_five_way  # noqa: E402
from config import (  # noqa: E402
    CURVE_COLUMNS,
    N_REAL_USERS,
    curves_path,
    dump_dir,
    target_item_weights_path,
    targets_path,
)
from roles import load_role_lookup, role_ids_for_features  # noqa: E402
from target_item_weights import TargetItemWeightRecorder, snapshot_target_items  # noqa: E402
from target_metrics import print_metrics, target_hit_rates, write_metrics  # noqa: E402
from curve_recorder import CurveRecorder  # noqa: E402
from weight_ops import fuse_with_switches, get_based_weight, production_get_weight  # noqa: E402

parser = argparse.ArgumentParser(description="GMF profile-pollution evaluation")
parser.add_argument("--model", type=str, default="GMF")
parser.add_argument("--dataset", type=str, default="movielens")
parser.add_argument("--data_path", type=str, default="../data/movielens/")
parser.add_argument("--setting", type=str, default="Full", choices=["Vanilla", "Full"])
parser.add_argument("--floor_ratio", type=float, default=0.3)
parser.add_argument("--factor_upper", type=float, default=0.25)
parser.add_argument("--seed", type=int, default=123456789)
parser.add_argument("--epoch_eval", type=int, default=10)
parser.add_argument("--top_k", type=int, nargs="+", default=[50, 100])
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--eval_interval", type=int, default=5)
parser.add_argument("--patience", type=int, default=6)
parser.add_argument("--gpu_id", type=int, default=-1)
parser.add_argument("--gpu_sort", type=str, default="mem", choices=["None", "mem", "util"])
parser.add_argument("--metrics_out", type=str, default="")
parser.add_argument("--n_real_users", type=int, default=N_REAL_USERS)
parser.add_argument("--targets_json", type=str, default="")
parser.add_argument("--stem", type=str, default="")
args = parser.parse_args()

model_name = "NeuMF-end" if args.model in ("NeuMF", "NeuMF-end") else args.model
dump_model_tag = "NeuMF" if model_name.startswith("NeuMF") else model_name
data_path = args.data_path if args.data_path.endswith("/") else args.data_path + "/"
if args.stem:
    stem = args.stem
else:
    stem = os.path.basename(os.path.normpath(data_path))
    if stem == "movielens" or "data/movielens" in os.path.normpath(data_path):
        stem = "no_attack"

print("[Profile-Pollution-Toolkit] GMF", args, "model_name=", model_name, "stem=", stem)

if args.gpu_id >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
else:
    gpu_ids = (
        get_gpu_id_str_sorted()
        if args.gpu_sort == "None"
        else get_gpu_id_str_sorted(strategy=args.gpu_sort)
    )
    if not gpu_ids:
        raise RuntimeError("Failed to get GPU IDs")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids.split(",")[0]

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):
    np.random.seed(args.seed + worker_id)


class NCFDataWithIndex(data_utils.NCFData):
    def __getitem__(self, idx):
        user, item, label, is_not_noisy = super().__getitem__(idx)
        return user, item, label, is_not_noisy, idx


(
    train_features_list,
    valid_features_list,
    test_pos,
    valid_pos,
    train_valid_pos,
    users_num,
    items_num,
    train_mat,
    train_valid_mat,
    train_data_noisy,
) = data_utils.load_all(args.dataset, data_path)

n_real = min(args.n_real_users, users_num)
valid_pos = {u: v for u, v in valid_pos.items() if u < n_real}
test_pos = {u: v for u, v in test_pos.items() if u < n_real}

tj = args.targets_json or targets_path()
with open(tj, "r", encoding="utf-8") as f:
    targets = [int(x) for x in json.load(f)["item_ids"]]

role_lookup = load_role_lookup(data_path.rstrip("/"))
role_per_pos = role_ids_for_features(train_features_list, train_data_noisy, role_lookup)

train_dataset = NCFDataWithIndex(
    features=train_features_list,
    items_num=items_num,
    train_mat=train_mat,
    neg_num_per_pos=1,
    phase=0,
    is_not_noisy=train_data_noisy,
)
train_loader = data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    worker_init_fn=worker_init_fn,
)

ncf_model = model.NCF(
    users_num=users_num,
    items_num=items_num,
    factors_num=32,
    layers_num=3,
    dropout=0.0,
    model_name=model_name,
    GMF_model=None,
    MLP_model=None,
)
ncf_model.cuda()
device = torch.device("cuda")
BCE_loss = nn.BCEWithLogitsLoss(reduction="none")
optimizer = optim.Adam(ncf_model.parameters(), lr=0.001)
train_mat_dense = torch.tensor(train_mat.toarray()).cpu()
train_valid_mat_dense = torch.tensor(train_valid_mat.toarray()).cpu()

num_train = len(train_features_list)
expanded_len = (1 + train_dataset.neg_num_per_pos) * num_train
w_final = torch.ones(expanded_len, device=device, dtype=torch.float32)
w_base_store = torch.ones(expanded_len, device=device, dtype=torch.float32)
w_user_store = torch.ones(expanded_len, device=device, dtype=torch.float32)
w_item_store = torch.ones(expanded_len, device=device, dtype=torch.float32)
item_entity_w = torch.ones(items_num, device=device, dtype=torch.float32)
train_user_interaction_cnts = torch.zeros(users_num, device=device, dtype=torch.float32)
train_item_interaction_cnts = torch.zeros(items_num, device=device, dtype=torch.float32)
role_pos_t = torch.tensor(role_per_pos, dtype=torch.long, device=device)

out_dir = dump_dir(dump_model_tag, stem, args.setting)
os.makedirs(out_dir, exist_ok=True)
recorder = CurveRecorder(curves_path(dump_model_tag, stem, args.setting), columns=CURVE_COLUMNS)
item_rec = TargetItemWeightRecorder(target_item_weights_path(dump_model_tag, stem, args.setting))

print(
    "data loaded! users_num:{}, items_num:{} train_data_len:{} test_user_num:{}".format(
        users_num, items_num, len(train_features_list), len(test_pos)
    )
)

patience_count = 0
best_recall = 0.0
best_epoch = None
best_model_state = None


def eval_model(model_, valid_pos_, mat, epoch):
    global best_recall, patience_count, best_epoch, best_model_state
    top_k = args.top_k
    model_.eval()
    predictedIndices = []
    GroundTruth = []
    users_in_valid = [u for u in valid_pos_ if u < n_real]
    for users_valid in toolz.partition_all(15, users_in_valid):
        users_valid = list(users_valid)
        GroundTruth.extend([valid_pos_[u] for u in users_valid])
        users_valid_torch = torch.tensor(users_valid).repeat_interleave(items_num).cuda()
        items_full = torch.tensor([i for i in range(items_num)]).repeat(len(users_valid)).cuda()
        prediction = model_(users_valid_torch, items_full)
        _, indices = torch.topk(
            prediction.view(len(users_valid), -1) + mat[users_valid].cuda() * -9999,
            max(top_k),
        )
        predictedIndices.extend(indices.cpu().numpy().tolist())
    precision, recall, NDCG, MRR = evaluate.compute_acc(GroundTruth, predictedIndices, top_k)
    epoch_recall = recall[0]
    print("################### EVAL ######################")
    print(f"Recall:{recall} NDCG: {NDCG}")
    if epoch_recall > best_recall:
        best_recall = epoch_recall
        best_epoch = epoch
        patience_count = 0
        best_model_state = {k: v.cpu().clone() for k, v in model_.state_dict().items()}
        print(f"New best model found! Recall: {best_recall:.4f} at epoch {epoch}")
    else:
        patience_count += 1


def rank_real_users(model_, mat, k):
    model_.eval()
    user_ids = list(range(n_real))
    ranks = []
    with torch.no_grad():
        for batch in toolz.partition_all(15, user_ids):
            batch = list(batch)
            users_t = torch.tensor(batch).repeat_interleave(items_num).cuda()
            items_full = torch.tensor([i for i in range(items_num)]).repeat(len(batch)).cuda()
            prediction = model_(users_t, items_full)
            _, indices = torch.topk(
                prediction.view(len(batch), -1) + mat[batch].cuda() * -9999,
                k,
            )
            ranks.extend(indices.cpu().numpy().tolist())
    return user_ids, ranks


t0 = time.time()
for epoch in range(1000):
    ncf_model.train()
    train_loss = 0.0
    train_loader.dataset.neg_sample()
    ui_expanded = torch.tensor(train_loader.dataset.features_fill, dtype=torch.long, device=device)
    u_all = ui_expanded[:, 0]
    i_all = ui_expanded[:, 1]
    train_user_interaction_cnts.zero_()
    train_item_interaction_cnts.zero_()
    user_epoch_loss_sum = torch.zeros(users_num, device=device, dtype=torch.float32)
    item_epoch_loss_sum = torch.zeros(items_num, device=device, dtype=torch.float32)
    all_bce_losses = []
    all_batch_indices = []
    loss_pos_list = []
    idx_pos_list = []

    for user, item, label, is_not_noisy, batch_idx in train_loader:
        user = user.long().to(device, non_blocking=True)
        item = item.long().to(device, non_blocking=True)
        label = label.float().to(device, non_blocking=True)
        batch_idx = batch_idx.long().to(device, non_blocking=True)
        ones = torch.ones_like(label)
        train_user_interaction_cnts.scatter_add_(0, user, ones)
        train_item_interaction_cnts.scatter_add_(0, item, ones)
        optimizer.zero_grad()
        prediction = ncf_model(user, item)
        loss = BCE_loss(prediction, label)
        with torch.no_grad():
            per_sample_loss = loss.detach()
            user_epoch_loss_sum.scatter_add_(0, user, per_sample_loss)
            item_epoch_loss_sum.scatter_add_(0, item, per_sample_loss)
            all_bce_losses.append(per_sample_loss)
            all_batch_indices.append(batch_idx)
            pos_mask = label > 0.5
            if pos_mask.any():
                loss_pos_list.append(per_sample_loss[pos_mask])
                idx_pos_list.append(batch_idx[pos_mask])
        if args.setting == "Vanilla" or epoch <= 1:
            mul_factor = torch.ones_like(loss)
        else:
            mul_factor = w_final[batch_idx]
        batch_loss = torch.mean(mul_factor * loss)
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss.item()

    with torch.no_grad():
        loss_pos = torch.cat(loss_pos_list)
        idx_pos = torch.cat(idx_pos_list)
        role_batch = role_pos_t[idx_pos]
        pos_items = torch.tensor(train_features_list, dtype=torch.long, device=device)[idx_pos, 1]
        if epoch <= 1 or args.setting == "Vanilla":
            wf = torch.ones_like(loss_pos)
            wb = torch.ones_like(loss_pos)
            wu = torch.ones_like(loss_pos)
            wi = torch.ones_like(loss_pos)
            ent = torch.ones(items_num, device=device, dtype=torch.float32)
        else:
            wf = w_final[idx_pos]
            wb = w_base_store[idx_pos]
            wu = w_user_store[idx_pos]
            wi = w_item_store[idx_pos]
            ent = item_entity_w
        stats = aggregate_five_way(
            loss=loss_pos,
            w_final=wf,
            w_base=wb,
            w_user=wu,
            w_item=wi,
            role_id=role_batch,
        )
        recorder.add(epoch, stats)
        item_rec.add_rows(
            epoch,
            snapshot_target_items(
                target_ids=targets,
                item_entity_w=ent,
                items=pos_items,
                w_final=wf,
                role_id=role_batch,
            ),
        )
    print(
        "epoch: {}, loss:{}, loss_target={}, w_target={}".format(
            epoch, train_loss, stats["loss_target"], stats["w_final_target"]
        )
    )
    if epoch >= args.epoch_eval and epoch % args.eval_interval == 0:
        eval_model(ncf_model, valid_pos, train_mat_dense, epoch)
    if patience_count == args.patience:
        break

    with torch.no_grad():
        if args.setting == "Vanilla":
            w_final.fill_(1.0)
            w_base_store.fill_(1.0)
            w_user_store.fill_(1.0)
            w_item_store.fill_(1.0)
            item_entity_w.fill_(1.0)
        else:
            user_avg_loss = user_epoch_loss_sum / train_user_interaction_cnts.clamp(min=1.0)
            item_avg_loss = item_epoch_loss_sum / train_item_interaction_cnts.clamp(min=1.0)
            user_fact = production_get_weight(
                user_avg_loss, args.factor_upper, train_user_interaction_cnts, args.floor_ratio
            )
            item_fact = production_get_weight(
                item_avg_loss, args.factor_upper, train_item_interaction_cnts, args.floor_ratio
            )
            all_loss = torch.cat(all_bce_losses)
            all_idx_tensor = torch.cat(all_batch_indices)
            cdf_weights = get_based_weight(-all_loss)
            w_base_per_idx = torch.zeros(expanded_len, device=device, dtype=torch.float32)
            w_base_per_idx[all_idx_tensor] = cdf_weights
            w_f, w_b, w_u, w_i = fuse_with_switches(
                w_base=w_base_per_idx,
                w_user=user_fact,
                w_item=item_fact,
                u_indices=u_all,
                i_indices=i_all,
                use_bw=True,
                use_uf=True,
                use_if=True,
            )
            w_final.copy_(w_f)
            w_base_store.copy_(w_b)
            w_user_store.copy_(w_u)
            w_item_store.copy_(w_i)
            item_entity_w.copy_(item_fact)

print("############################## Training End. ##############################")
print(f"Whole Cost time: {(time.time() - t0) / 60:.2f} min")
if best_model_state is not None:
    ncf_model.load_state_dict(best_model_state)
    ncf_model.cuda()
    print(f"Best model found in epoch {best_epoch}")
else:
    print("No best model captured during eval; using current model for testing.")

ncf_model.eval()
predictedIndices = []
GroundTruth = []
users_in_test = [u for u in test_pos if u < n_real]
for users_test in toolz.partition_all(15, users_in_test):
    users_test = list(users_test)
    GroundTruth.extend([test_pos[u] for u in users_test])
    users_test_torch = torch.tensor(users_test).repeat_interleave(items_num).cuda()
    items_full = torch.tensor([i for i in range(items_num)]).repeat(len(users_test)).cuda()
    prediction = ncf_model(users_test_torch, items_full)
    _, indices = torch.topk(
        prediction.view(len(users_test), -1) + train_valid_mat_dense[users_test].cuda() * -9999,
        max(args.top_k),
    )
    predictedIndices.extend(indices.cpu().numpy().tolist())
precision, recall, NDCG, MRR = evaluate.compute_acc(GroundTruth, predictedIndices, args.top_k)

user_ids, ranks = rank_real_users(ncf_model, train_valid_mat_dense, max(50, max(args.top_k)))
consumed = {}
for u in range(n_real):
    consumed[u] = set(train_valid_pos.get(u, [])) | set(test_pos.get(u, []))
target_rows = target_hit_rates(
    user_ids=user_ids,
    topk_lists=ranks,
    targets=targets,
    consumed=consumed,
    ks=(10, 20, 50),
    n_real_users=n_real,
)
print_metrics(recall, NDCG, args.top_k, target_rows)
if args.metrics_out:
    write_metrics(
        args.metrics_out,
        recall=recall,
        ndcg=NDCG,
        overall_ks=args.top_k,
        target_rows=target_rows,
        extra={
            "best_epoch": best_epoch,
            "setting": args.setting,
            "model": args.model,
            "stem": stem,
        },
    )
    print(f"metrics: {args.metrics_out}")
print(f"curves: {recorder.path} rows={len(recorder.rows)}")
print(f"target_item_weights: {item_rec.path} rows={len(item_rec.rows)}")
