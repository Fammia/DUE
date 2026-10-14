"""LightGCN profile-pollution evaluation: ranking, target exposure, and weight traces."""

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
import torch.nn.functional as F
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

parser = argparse.ArgumentParser(description="LightGCN profile-pollution evaluation")
parser.add_argument("--dataset", type=str, default="movielens")
parser.add_argument("--data_path", type=str, default="../data/movielens/")
parser.add_argument("--setting", type=str, default="Full", choices=["Vanilla", "Full"])
parser.add_argument("--floor_ratio", type=float, default=0.1)
parser.add_argument("--factor_upper", type=float, default=0.95)
parser.add_argument("--seed", type=int, default=123456789)
parser.add_argument("--epoch_eval", type=int, default=10)
parser.add_argument("--top_k", type=int, nargs="+", default=[50, 100])
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--eval_interval", type=int, default=1)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--gpu_id", type=int, default=-1)
parser.add_argument("--gpu_sort", type=str, default="mem", choices=["None", "mem", "util"])
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--layers", type=int, default=2)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--l2_reg", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--metrics_out", type=str, default="")
parser.add_argument("--n_real_users", type=int, default=N_REAL_USERS)
parser.add_argument("--targets_json", type=str, default="")
parser.add_argument("--stem", type=str, default="")
args = parser.parse_args()

data_path = args.data_path if args.data_path.endswith("/") else args.data_path + "/"
if args.stem:
    stem = args.stem
else:
    stem = os.path.basename(os.path.normpath(data_path))
    if stem == "movielens" or "data/movielens" in os.path.normpath(data_path):
        stem = "no_attack"

print("[Profile-Pollution-Toolkit] LightGCN", args, "stem=", stem)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):
    np.random.seed(args.seed + worker_id)


class BPRDataWithIndex(data_utils.BPRData):
    def __getitem__(self, idx):
        u, i_pos = self.features[idx][0], self.features[idx][1]
        i_neg = np.random.randint(self.items_num)
        while i_neg in self.train_user_pos_set.get(u, set()):
            i_neg = np.random.randint(self.items_num)
        return u, i_pos, i_neg, idx


def bpr_reg_loss(model_, users, pos_items, neg_items):
    users_emb_ego = model_.user_embedding(users)
    pos_emb_ego = model_.item_embedding(pos_items)
    neg_emb_ego = model_.item_embedding(neg_items)
    return (
        users_emb_ego.norm(2).pow(2)
        + pos_emb_ego.norm(2).pow(2)
        + neg_emb_ego.norm(2).pow(2)
    ) / (2.0 * len(users))


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
    train_user_pos_set,
) = data_utils.load_all(args.dataset, data_path)

n_real = min(args.n_real_users, users_num)
valid_pos = {u: v for u, v in valid_pos.items() if u < n_real}
test_pos = {u: v for u, v in test_pos.items() if u < n_real}

tj = args.targets_json or targets_path()
with open(tj, "r", encoding="utf-8") as f:
    targets = [int(x) for x in json.load(f)["item_ids"]]

role_lookup = load_role_lookup(data_path.rstrip("/"))
role_per_pos = role_ids_for_features(train_features_list, train_data_noisy, role_lookup)
role_t = torch.tensor(role_per_pos, dtype=torch.long, device=device)

num_train = len(train_features_list)
train_ui_tensor = torch.tensor(train_features_list, dtype=torch.long, device=device)
train_dataset = BPRDataWithIndex(
    features=train_features_list,
    items_num=items_num,
    train_user_pos_set=train_user_pos_set,
)
train_loader = data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    worker_init_fn=worker_init_fn,
)

train_mat_dense = torch.tensor(train_mat.toarray(), dtype=torch.float32).cpu()
train_valid_mat_dense = torch.tensor(train_valid_mat.toarray(), dtype=torch.float32).cpu()
print(
    "data loaded! users_num:{}, items_num:{} train_data_len:{} test_user_num:{}".format(
        users_num, items_num, len(train_features_list), len(test_pos)
    )
)

norm_adj = model.build_norm_adj(train_mat, users_num, items_num).to(device)
lightgcn_model = model.LightGCN(
    users_num=users_num,
    items_num=items_num,
    embedding_dim=args.embedding_dim,
    layers=args.layers,
    norm_adj=norm_adj,
).to(device)
optimizer = optim.Adam(lightgcn_model.parameters(), lr=args.lr)

patience_count = 0
best_recall = 0.0
best_epoch = None
best_model_state = None


def eval_phase(model_, valid_data_pos, mask_mat, epoch):
    global best_recall, patience_count, best_epoch, best_model_state
    precision, recall, NDCG, MRR = evaluate.eval_all_rank(
        model=model_,
        eval_pos=valid_data_pos,
        mask_mat=mask_mat,
        items_num=items_num,
        top_k=args.top_k,
        device=device,
    )
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


w_final = torch.ones(num_train, device=device, dtype=torch.float32)
w_base_store = torch.ones(num_train, device=device, dtype=torch.float32)
w_user_store = torch.ones(num_train, device=device, dtype=torch.float32)
w_item_store = torch.ones(num_train, device=device, dtype=torch.float32)
item_entity_w = torch.ones(items_num, device=device, dtype=torch.float32)
user_loss_sum = torch.zeros(users_num, device=device, dtype=torch.float32)
user_loss_count = torch.zeros(users_num, device=device, dtype=torch.float32)
item_loss_sum = torch.zeros(items_num, device=device, dtype=torch.float32)
item_loss_count = torch.zeros(items_num, device=device, dtype=torch.float32)

out_dir = dump_dir("LightGCN", stem, args.setting)
os.makedirs(out_dir, exist_ok=True)
recorder = CurveRecorder(curves_path("LightGCN", stem, args.setting), columns=CURVE_COLUMNS)
item_rec = TargetItemWeightRecorder(target_item_weights_path("LightGCN", stem, args.setting))

t0 = time.time()
for epoch in range(args.epochs):
    lightgcn_model.train()
    train_loss = 0.0
    user_loss_sum.zero_()
    user_loss_count.zero_()
    item_loss_sum.zero_()
    item_loss_count.zero_()
    all_bpr_losses = []
    all_batch_indices = []
    loss_list = []
    idx_list = []

    for user, item_pos, item_neg, batch_idx in train_loader:
        user = user.long().to(device, non_blocking=True)
        item_pos = item_pos.long().to(device, non_blocking=True)
        item_neg = item_neg.long().to(device, non_blocking=True)
        batch_idx = batch_idx.long().to(device, non_blocking=True)
        user_emb_all, item_emb_all = lightgcn_model.get_all_embeddings()
        users_emb = user_emb_all[user]
        pos_scores = torch.sum(users_emb * item_emb_all[item_pos], dim=1)
        neg_scores = torch.sum(users_emb * item_emb_all[item_neg], dim=1)
        bpr_loss_raw = -F.logsigmoid(pos_scores - neg_scores)
        reg_loss = bpr_reg_loss(lightgcn_model, user, item_pos, item_neg)
        with torch.no_grad():
            per_sample_bpr = bpr_loss_raw.detach()
            ones = torch.ones_like(per_sample_bpr)
            user_loss_sum.scatter_add_(0, user, per_sample_bpr)
            user_loss_count.scatter_add_(0, user, ones)
            item_loss_sum.scatter_add_(0, item_pos, per_sample_bpr)
            item_loss_count.scatter_add_(0, item_pos, ones)
            all_bpr_losses.append(per_sample_bpr)
            all_batch_indices.append(batch_idx)
            loss_list.append(per_sample_bpr)
            idx_list.append(batch_idx)
        if args.setting == "Vanilla" or epoch == 0:
            mul_factor = torch.ones_like(bpr_loss_raw)
        else:
            mul_factor = w_final[batch_idx]
        batch_loss = torch.mean(mul_factor * bpr_loss_raw) + args.l2_reg * reg_loss
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss.item()

    with torch.no_grad():
        loss_all = torch.cat(loss_list)
        idx_all = torch.cat(idx_list)
        role_batch = role_t[idx_all]
        pos_items = train_ui_tensor[idx_all, 1]
        if epoch == 0 or args.setting == "Vanilla":
            wf = torch.ones_like(loss_all)
            wb = torch.ones_like(loss_all)
            wu = torch.ones_like(loss_all)
            wi = torch.ones_like(loss_all)
            ent = torch.ones(items_num, device=device, dtype=torch.float32)
        else:
            wf = w_final[idx_all]
            wb = w_base_store[idx_all]
            wu = w_user_store[idx_all]
            wi = w_item_store[idx_all]
            ent = item_entity_w
        stats = aggregate_five_way(
            loss=loss_all,
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
        eval_phase(lightgcn_model, valid_pos, train_mat_dense, epoch)
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
            user_avg_loss = user_loss_sum / user_loss_count.clamp(min=1.0)
            item_avg_loss = item_loss_sum / item_loss_count.clamp(min=1.0)
            user_fact = production_get_weight(
                user_avg_loss, args.factor_upper, user_loss_count, args.floor_ratio
            )
            item_fact = production_get_weight(
                item_avg_loss, args.factor_upper, item_loss_count, args.floor_ratio
            )
            all_bpr = torch.cat(all_bpr_losses)
            all_idx_tensor = torch.cat(all_batch_indices)
            cdf = get_based_weight(-all_bpr)
            w_base_per_idx = torch.zeros(num_train, device=device, dtype=torch.float32)
            w_base_per_idx[all_idx_tensor] = cdf
            w_f, w_b, w_u, w_i = fuse_with_switches(
                w_base=w_base_per_idx,
                w_user=user_fact,
                w_item=item_fact,
                u_indices=train_ui_tensor[:, 0],
                i_indices=train_ui_tensor[:, 1],
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
    lightgcn_model.load_state_dict(best_model_state)
    print(f"Best model found in epoch {best_epoch}")
else:
    print("No best model captured during eval; using current model for testing.")
lightgcn_model.to(device)

precision, recall, NDCG, MRR = evaluate.eval_all_rank(
    model=lightgcn_model,
    eval_pos=test_pos,
    mask_mat=train_valid_mat_dense,
    items_num=items_num,
    top_k=args.top_k,
    device=device,
)

user_ids = list(range(n_real))
k_rank = max(50, max(args.top_k))
lightgcn_model.eval()
ranks = []
with torch.no_grad():
    for batch in toolz.partition_all(1024, user_ids):
        batch = list(batch)
        users_t = torch.tensor(batch, dtype=torch.long, device=device)
        prediction = lightgcn_model.getUsersRating(users_t)
        prediction = prediction + train_valid_mat_dense[batch].to(device) * -9999
        _, indices = torch.topk(prediction, k_rank)
        ranks.extend(indices.cpu().numpy().tolist())

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
            "model": "LightGCN",
            "stem": stem,
        },
    )
    print(f"metrics: {args.metrics_out}")
print(f"curves: {recorder.path} rows={len(recorder.rows)}")
print(f"target_item_weights: {item_rec.path} rows={len(item_rec.rows)}")
