import argparse
import os
import sys
import time
import random
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import model
import data_utils
import evaluate

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from common.gpu_utils import get_gpu_id_str_sorted


parser = argparse.ArgumentParser(description="LightGCN + DUE training")

parser.add_argument("--dataset", type=str, help="dataset used for training, options: amazon_book, yelp, movielens", default="movielens")
parser.add_argument("--floor_ratio", type=float, default=0.0, help="minimum entity factor as a fraction of factor_upper")
parser.add_argument("--factor_upper", type=float, default=1.5, help="entity weight upper bound (beta)")
parser.add_argument("--seed", type=int, help="seed for reproducibility", default=123456789)
parser.add_argument("--epoch_eval", type=int, default=10, help="epoch to start evaluation")
parser.add_argument("--top_k", type=int, nargs="+", default=[50, 100], help="compute metric @topk")
parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
parser.add_argument("--eval_interval", type=int, default=1, help="Interval for evaluation")
parser.add_argument("--patience", type=int, default=10, help="patience")
parser.add_argument("--gpu_id", type=int, default=-1, help="gpu_id")
parser.add_argument("--gpu_sort", type=str, default="None", choices=["None", "mem", "util"], help="auto GPU pick when gpu_id<0: None=use gpu_utils default strategy, mem=free VRAM (desc), util=GPU compute util (asc, lowest first); default None")

parser.add_argument("--embedding_dim", type=int, default=64, help="embedding dim")
parser.add_argument("--layers", type=int, default=2, help="graph layers")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--l2_reg", type=float, default=1e-4, help="l2 regularization")
parser.add_argument("--epochs", type=int, default=1000, help="training epochs")
args = parser.parse_args()

print("[DUE] LightGCN training and evaluation")
assert args.dataset in ["movielens", "yelp", "amazon_book"]
print(f"args: {args}")


if args.gpu_id >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"Using manually specified GPU: {args.gpu_id}")
else:
    gpu_ids = get_gpu_id_str_sorted() if args.gpu_sort == "None" else get_gpu_id_str_sorted(strategy=args.gpu_sort)
    if not gpu_ids:
        raise RuntimeError("Failed to get GPU IDs from nvidia-smi.")
    selected_gpu = gpu_ids.split(",")[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpu
    print(f"Using automatically selected GPU: {selected_gpu}")
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
    """Same as BPRData but returns dataset index for aligning per-sample weights."""

    def __getitem__(self, idx):
        u, i_pos = self.features[idx][0], self.features[idx][1]
        i_neg = np.random.randint(self.items_num)
        while i_neg in self.train_user_pos_set.get(u, set()):
            i_neg = np.random.randint(self.items_num)
        return u, i_pos, i_neg, idx


def bpr_reg_loss(model_, users, pos_items, neg_items):
    """L2 on ego embeddings of sampled triplets (same as main_vanilla bpr_loss)."""
    users_emb_ego = model_.user_embedding(users)
    pos_emb_ego = model_.item_embedding(pos_items)
    neg_emb_ego = model_.item_embedding(neg_items)
    reg_loss = (
        users_emb_ego.norm(2).pow(2)
        + pos_emb_ego.norm(2).pow(2)
        + neg_emb_ego.norm(2).pow(2)
    ) / (2.0 * len(users))
    return reg_loss


def get_weight(avg_loss, startValue, interaction_cnts=None, floor_ratio=0.0):
    """Map entity losses to support-gated asymmetric reliability factors.

    Lower loss ranks receive higher factors in [startValue * floor_ratio,
    startValue]. With support counts, shrink toward the interval midpoint;
    the gate accounts for support, extremity, and suppression direction.
    """
    effective_end = startValue * floor_ratio
    sorted_indices = torch.argsort(avg_loss)
    n = avg_loss.numel()
    weights = torch.linspace(
        startValue, effective_end, steps=n,
        device=avg_loss.device, dtype=avg_loss.dtype,
    )
    result = torch.zeros_like(avg_loss)
    result[sorted_indices] = weights

    if interaction_cnts is not None:
        neutral = (startValue + effective_end) * 0.5
        prior_n = interaction_cnts.median().clamp(min=1.0)
        half_range = abs(startValue - effective_end) * 0.5

        if half_range > 1e-8:
            extremity = (result - neutral).abs() / half_range
            suppression_depth = ((neutral - result) / half_range).clamp(min=0.0)
            risk = extremity * (1.0 + suppression_depth)
        else:
            risk = torch.zeros_like(result)

        effective_prior = prior_n * (1.0 + risk)
        raw_confidence = interaction_cnts / (interaction_cnts + effective_prior)
        confidence = raw_confidence.sqrt()
        result = confidence * result + (1.0 - confidence) * neutral

    return result


def get_based_weight(loss_values):
    """
    ECDF-based interaction weights (EARD base weight).
    Per-sample BPR losses; pass -loss for higher weight on lower loss.
    """
    if loss_values.numel() == 0:
        return torch.ones_like(loss_values)

    sorted_losses, _ = torch.sort(loss_values)
    ranks = torch.searchsorted(sorted_losses, loss_values, right=True).float()
    n_samples = loss_values.numel()
    cdf_weights = (ranks - 0.5) / n_samples
    return cdf_weights


########################### Weight Fusion #####################################
def fuse_weights(w_base, w_user, w_item, u_indices, i_indices):
    """Multiply interaction factors by the indexed user and item factors."""
    w_u_per_sample = w_user[u_indices]
    w_i_per_sample = w_item[i_indices]
    w_final = w_base * w_u_per_sample * w_i_per_sample
    return w_final


data_path = f"../data/{args.dataset}/"
model_path = "./models/"
os.makedirs(model_path, exist_ok=True)

############################## PREPARE DATASET ##########################
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
) = data_utils.load_all(f"{args.dataset}", data_path)

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

########################### CREATE MODEL #################################
norm_adj = model.build_norm_adj(train_mat, users_num, items_num).to(device)
lightgcn_model = model.LightGCN(
    users_num=users_num,
    items_num=items_num,
    embedding_dim=args.embedding_dim,
    layers=args.layers,
    norm_adj=norm_adj,
).to(device)
optimizer = optim.Adam(lightgcn_model.parameters(), lr=args.lr)

########################### Eval / Test #####################################
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


def test_phase(model_, test_data_pos, mask_mat):
    precision, recall, NDCG, MRR = evaluate.eval_all_rank(
        model=model_,
        eval_pos=test_data_pos,
        mask_mat=mask_mat,
        items_num=items_num,
        top_k=args.top_k,
        device=device,
    )
    print("################### TEST ######################")
    print("Recall {:.4f}-{:.4f}".format(recall[0], recall[1]))
    print("NDCG {:.4f}-{:.4f}".format(NDCG[0], NDCG[1]))


########################### Training Phase #####################################
w_final = torch.ones(num_train, device=device, dtype=torch.float32)

startTime = time.time()

user_loss_sum = torch.zeros(users_num, device=device, dtype=torch.float32)
user_loss_count = torch.zeros(users_num, device=device, dtype=torch.float32)
item_loss_sum = torch.zeros(items_num, device=device, dtype=torch.float32)
item_loss_count = torch.zeros(items_num, device=device, dtype=torch.float32)

for epoch in range(args.epochs):
    lightgcn_model.train()
    train_loss = 0.0

    user_loss_sum.zero_()
    user_loss_count.zero_()
    item_loss_sum.zero_()
    item_loss_count.zero_()

    all_bpr_losses = []
    all_batch_indices = []

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

        if epoch == 0:
            mul_factor = torch.ones_like(bpr_loss_raw)
        else:
            mul_factor = w_final[batch_idx]

        batch_loss = torch.mean(mul_factor * bpr_loss_raw) + args.l2_reg * reg_loss

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss.item()

    print("epoch: {}, loss:{}".format(epoch, train_loss))

    if epoch >= args.epoch_eval and epoch % args.eval_interval == 0:
        eval_phase(
            model_=lightgcn_model,
            valid_data_pos=valid_pos,
            mask_mat=train_mat_dense,
            epoch=epoch,
        )

    if patience_count == args.patience:
        break

    with torch.no_grad():
        user_avg_loss = user_loss_sum / user_loss_count.clamp(min=1.0)
        item_avg_loss = item_loss_sum / item_loss_count.clamp(min=1.0)

        user_fact = get_weight(
            user_avg_loss,
            startValue=args.factor_upper,
            interaction_cnts=user_loss_count,
            floor_ratio=args.floor_ratio,
        )
        item_fact = get_weight(
            item_avg_loss,
            startValue=args.factor_upper,
            interaction_cnts=item_loss_count,
            floor_ratio=args.floor_ratio,
        )

        all_bpr = torch.cat(all_bpr_losses)
        all_idx_tensor = torch.cat(all_batch_indices)
        cdf = get_based_weight(-all_bpr)

        w_base_per_idx = torch.zeros(num_train, device=device, dtype=torch.float32)
        w_base_per_idx[all_idx_tensor] = cdf

        u_all = train_ui_tensor[:, 0]
        i_all = train_ui_tensor[:, 1]
        w_final.copy_(fuse_weights(
            w_base=w_base_per_idx,
            w_user=user_fact,
            w_item=item_fact,
            u_indices=u_all,
            i_indices=i_all,
        ))

    all_bpr_losses.clear()
    all_batch_indices.clear()


print("############################## Training End. ##############################")
print(f"Whole Cost time: {(time.time()-startTime) / 60:.2f} min")
print(f"args: {args}")

model_save_path = os.path.join(
    model_path,
    f"LightGCN_{args.dataset}_DUE_ratio-{args.floor_ratio}_upper-{args.factor_upper}.pth",
)
if best_model_state is not None:
    torch.save(best_model_state, model_save_path)
    print(f"Best model found in epoch {best_epoch}")
    print(f"Best model saved to: {model_save_path}")
    lightgcn_model.load_state_dict(best_model_state)
else:
    print("No best model captured during eval; using current model for testing.")

lightgcn_model.to(device)
test_phase(lightgcn_model, test_pos, train_valid_mat_dense)
