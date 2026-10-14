import os
import sys
import time
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from tqdm import tqdm
import toolz
import model
import evaluate
import data_utils

import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from common.gpu_utils import get_gpu_id_str_sorted

parser = argparse.ArgumentParser(description="Training script")

parser.add_argument("--model", type=str, help="model used for training. options: GMF, NeuMF-end", default="NeuMF-end")
parser.add_argument("--dataset", type=str, help="dataset used for training, options: amazon_book, yelp, movielens", default="movielens")
parser.add_argument("--floor_ratio", type=float, default=0.0, help="minimum entity factor as a fraction of factor_upper")
parser.add_argument("--factor_upper", type=float, default=1.5, help="factor upper")

parser.add_argument("--seed", type=int, help="seed for reproducibility", default=123456789)
parser.add_argument("--epoch_eval", type=int, default=10, help="epoch to start evaluation")
parser.add_argument("--top_k", type=int, nargs="+", default=[50, 100], help="compute metric @topk")
parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
parser.add_argument("--eval_interval", type=int, default=1, help="Interval for evaluation")
parser.add_argument("--patience", type=int, default=10, help="patience")
parser.add_argument("--name", type=str, default="", help="name")
parser.add_argument("--gpu_id", type=int, default=-1, help="gpu_id")
parser.add_argument("--gpu_sort", type=str, default="None", choices=["None", "mem", "util"], help="auto GPU pick when gpu_id<0: None=use gpu_utils default strategy, mem=free VRAM (desc), util=GPU compute util (asc, lowest first); default None")
args = parser.parse_args()

print("[DUE] NCF training and evaluation")
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

torch.manual_seed(args.seed)  # cpu
torch.cuda.manual_seed(args.seed)  # gpu
np.random.seed(args.seed)  # numpy
random.seed(args.seed)  # random and transforms
cudnn.benchmark = True
torch.backends.cudnn.deterministic = True  # cudnn


def worker_init_fn(worker_id):
    np.random.seed(args.seed + worker_id)


class NCFDataWithIndex(data_utils.NCFData):
    """Training dataset: same as NCFData but returns sample index for EARD weights."""

    def __getitem__(self, idx):
        user, item, label, is_not_noisy = super().__getitem__(idx)
        return user, item, label, is_not_noisy, idx


data_path = f"../data/{args.dataset}/"
model_path = f"./models/"

# Create model save path
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
) = data_utils.load_all(f"{args.dataset}", data_path)

# construct the train and test datasets
train_dataset = NCFDataWithIndex(
    features=train_features_list,
    items_num=items_num,
    train_mat=train_mat,
    neg_num_per_pos=1,
    phase=0,
    is_not_noisy=train_data_noisy,
)

valid_dataset = data_utils.NCFData(
    features=valid_features_list,
    items_num=items_num,
    train_mat=train_mat,
    neg_num_per_pos=1,
    phase=1,
    is_not_noisy=None,
)

train_loader = data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    worker_init_fn=worker_init_fn,
)

valid_loader = data.DataLoader(
    valid_dataset,
    batch_size=2048,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    worker_init_fn=worker_init_fn,
)

print(
    "data loaded! users_num:{}, items_num:{} train_data_len:{} test_user_num:{}".format(
        users_num, items_num, len(train_features_list), len(test_pos)
    )
)

########################### CREATE MODEL #################################

ncf_model = model.NCF(
    users_num=users_num,
    items_num=items_num,
    factors_num=32,
    layers_num=3,
    dropout=0.0,
    model_name=f"{args.model}",
    GMF_model=None,
    MLP_model=None,
)

ncf_model.cuda()
device = torch.device("cuda")
BCE_loss = nn.BCEWithLogitsLoss(reduction="none")
optimizer = optim.Adam(ncf_model.parameters(), lr=0.001)
train_mat_dense = torch.tensor(train_mat.toarray()).cpu()
train_user_interaction_cnts = torch.zeros(users_num, device=device, dtype=torch.float32)
train_item_interaction_cnts = torch.zeros(items_num, device=device, dtype=torch.float32)

num_train = len(train_features_list)
expanded_len = (1 + train_dataset.neg_num_per_pos) * num_train
w_final = torch.ones(expanded_len, device=device, dtype=torch.float32)

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


########################### Temp Scaled Weight Base #####################################
def get_based_weight(loss_values):
    """
    Weight calculation based on Cumulative Distribution Function (CDF) - Sparse implementation
    Theoretical basis: Weight allocation based on the position of loss values in the empirical distribution

    Args:
        loss_values: List of loss values, already in -loss form

    Returns:
        User index list, item index list, weight value list
    """
    if loss_values.numel() == 0:
        return torch.ones_like(loss_values)

    # Sort loss values
    sorted_losses, _ = torch.sort(loss_values)

    # Calculate the position (rank) of each loss value in the sorted array
    ranks = torch.searchsorted(sorted_losses, loss_values, right=True).float()

    # Calculate CDF values using Weibull plotting position: (rank - 0.5) / n
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


########################### Eval Phase Definition #####################################
patience_count, best_hr = 0, 0  # Early stopping counter, best HR
best_recall = 0.0  # Best Recall
best_epoch = None  # Epoch corresponding to the best model
best_model_state = None  # Save the best model's state_dict in memory


def eval(model, valid_pos, mat, epoch):
    global best_recall, patience_count, best_epoch, best_model_state

    top_k = args.top_k
    model.eval()
    predictedIndices = []
    GroundTruth = []
    users_in_valid = list(valid_pos.keys())
    for users_valid in toolz.partition_all(15, users_in_valid):
        users_valid = list(users_valid)
        GroundTruth.extend([valid_pos[u] for u in users_valid])
        users_valid_torch = (
            torch.tensor(users_valid).repeat_interleave(items_num).cuda()
        )
        items_full = (
            torch.tensor([i for i in range(items_num)]).repeat(len(users_valid)).cuda()
        )
        prediction = model(users_valid_torch, items_full)
        _, indices = torch.topk(
            prediction.view(len(users_valid), -1) + mat[users_valid].cuda() * -9999,
            max(top_k),
        )
        indices = indices.cpu().numpy().tolist()
        predictedIndices.extend(indices)
    precision, recall, NDCG, MRR = evaluate.compute_acc(
        GroundTruth, predictedIndices, top_k
    )
    epoch_recall = recall[0]

    print("################### EVAL ######################")
    print(f"Recall:{recall} NDCG: {NDCG}")

    if epoch_recall > best_recall:
        best_recall = epoch_recall
        best_epoch = epoch
        patience_count = 0
        # Save the best model's state_dict in memory to avoid frequent disk I/O
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"New best model found! Recall: {best_recall:.4f} at epoch {epoch}")
    else:
        patience_count += 1


########################### Test Phase Definition #####################################
def test(model, test_data_pos, mat):
    top_k = args.top_k
    model.eval()
    predictedIndices = []
    GroundTruth = []
    users_in_test = list(test_data_pos.keys())
    for users_test in toolz.partition_all(15, users_in_test):
        users_test = list(users_test)
        GroundTruth.extend([test_data_pos[u] for u in users_test])
        users_test_torch = torch.tensor(users_test).repeat_interleave(items_num).cuda()
        items_full = (
            torch.tensor([i for i in range(items_num)]).repeat(len(users_test)).cuda()
        )
        prediction = model(users_test_torch, items_full)
        _, indices = torch.topk(
            prediction.view(len(users_test), -1) + mat[users_test].cuda() * -9999,
            max(top_k),
        )
        indices = indices.cpu().numpy().tolist()
        predictedIndices.extend(indices)
    precision, recall, NDCG, MRR = evaluate.compute_acc(
        GroundTruth, predictedIndices, top_k
    )

    print("################### TEST ######################")
    print("Recall {:.4f}-{:.4f}".format(recall[0], recall[1]))
    print("NDCG {:.4f}-{:.4f}".format(NDCG[0], NDCG[1]))


########################### Training Phase #####################################
startTime = time.time()
top_k = args.top_k

all_bce_losses = []
all_batch_indices = []

for epoch in range(1000):
    ncf_model.train()  # Enable dropout (if have).
    train_loss = 0.0

    train_loader.dataset.neg_sample()  # negative sampling is done here
    ui_expanded = torch.tensor(
        train_loader.dataset.features_fill, dtype=torch.long, device=device
    )
    u_all = ui_expanded[:, 0]
    i_all = ui_expanded[:, 1]

    train_user_interaction_cnts.zero_()
    train_item_interaction_cnts.zero_()

    user_epoch_loss_sum = torch.zeros(users_num, device=device, dtype=torch.float32)
    item_epoch_loss_sum = torch.zeros(items_num, device=device, dtype=torch.float32)

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

        if epoch <= 1:
            mul_factor = torch.ones_like(loss)
        else:
            mul_factor = w_final[batch_idx]

        batch_loss = torch.mean(mul_factor * loss)

        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss.item()

    print("epoch: {}, loss:{}".format(epoch, train_loss))

    if epoch >= args.epoch_eval and epoch % args.eval_interval == 0:
        eval(
            model=ncf_model,
            valid_pos=valid_pos,
            mat=train_mat_dense,
            epoch=epoch,
        )

    if patience_count == args.patience:
        break

    with torch.no_grad():
        user_avg_loss = user_epoch_loss_sum / train_user_interaction_cnts.clamp(min=1.0)
        item_avg_loss = item_epoch_loss_sum / train_item_interaction_cnts.clamp(min=1.0)

        user_fact = get_weight(
            avg_loss=user_avg_loss,
            startValue=args.factor_upper,
            interaction_cnts=train_user_interaction_cnts,
            floor_ratio=args.floor_ratio,
        )
        item_fact = get_weight(
            avg_loss=item_avg_loss,
            startValue=args.factor_upper,
            interaction_cnts=train_item_interaction_cnts,
            floor_ratio=args.floor_ratio,
        )

        all_loss = torch.cat(all_bce_losses)
        all_idx_tensor = torch.cat(all_batch_indices)
        cdf_weights = get_based_weight(-all_loss)

        w_base_per_idx = torch.zeros(expanded_len, device=device, dtype=torch.float32)
        w_base_per_idx[all_idx_tensor] = cdf_weights

        w_final.copy_(fuse_weights(
            w_base=w_base_per_idx,
            w_user=user_fact,
            w_item=item_fact,
            u_indices=u_all,
            i_indices=i_all,
        ))

    all_bce_losses.clear()
    all_batch_indices.clear()


print("############################## Training End. ##############################")

print(f"Whole Cost time: {(time.time()-startTime) / 60:.2f} min")
print(f"args: {args}")

model_save_path = os.path.join(
    model_path,
    f"{args.model}_{args.dataset}_ratio-{args.floor_ratio}_upper-{args.factor_upper}.pth",
)

if best_model_state is not None:
    torch.save(best_model_state, model_save_path)
    print(f"Best model found in epoch {best_epoch}")
    print(f"Best model saved to: {model_save_path}")

    ncf_model.load_state_dict(best_model_state)
    ncf_model.cuda()
    test_model = ncf_model

########################### Logs Phase #####################################

train_mat_dense = torch.tensor(train_valid_mat.toarray()).cpu()
test(test_model, test_pos, train_mat_dense)
