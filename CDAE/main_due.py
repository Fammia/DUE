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

import toolz
import model
import evaluate
import data_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from common.gpu_utils import get_gpu_id_str_sorted

parser = argparse.ArgumentParser(description="Training script")

parser.add_argument("--dataset", type=str, help="dataset used for training, options: amazon_book, yelp, movielens", default="movielens")
parser.add_argument("--floor_ratio", type=float, default=0.0, help="minimum entity factor as a fraction of factor_upper")
parser.add_argument("--factor_upper", type=float, default=1.5, help="factor upper")
parser.add_argument("--seed", type=int, help="seed for reproducibility", default=123456789)
parser.add_argument("--epoch_eval", type=int, default=10, help="epoch to start evaluation")
parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
parser.add_argument("--top_k", type=int, nargs="+", default=[50, 100], help="compute metrics @k")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")

parser.add_argument("--eval_interval", type=int, default=5, help="Interval for evaluation")
parser.add_argument("--patience", type=int, default=5, help="patience")
parser.add_argument("--gpu_id", type=int, default=-1, help="gpu_id")
parser.add_argument("--gpu_sort", type=str, default="None", choices=["None", "mem", "util"], help="auto GPU pick when gpu_id<0: None=use gpu_utils default strategy, mem=free VRAM (desc), util=GPU compute util (asc, lowest first); default None")
args = parser.parse_args()

print("[DUE] CDAE training and evaluation")
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


data_path = f"../data/{args.dataset}/"
model_path = f"./models/"

# Create directory to save models
os.makedirs(model_path, exist_ok=True)

############################## PREPARE DATASET ##########################

(
    train_features_list,
    valid_features_list,
    train_pos,
    valid_pos,
    test_pos,
    train_valid_pos,
    users_num,
    items_num,
    train_mat,
    valid_mat,
    train_data_noisy,
) = data_utils.load_all(f"{args.dataset}", data_path)


train_mat_dense = train_mat.toarray()
users_list = np.array([i for i in range(users_num)])
train_dataset = data_utils.DenseMatrixUsers(users_list, train_mat_dense)
train_loader = data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
)

valid_mat_dense = valid_mat.toarray()
valid_dataset = data_utils.DenseMatrixUsers(users_list, valid_mat_dense)
valid_loader = data.DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
)

########################### CREATE MODEL #################################

cdae_model = model.CDAE(
    users_num=users_num, items_num=items_num, hidden_dim=32, dropout=0.2
)
cdae_model.cuda()
BCE_loss = nn.BCEWithLogitsLoss(reduction="none")
num_ns = 1  # negative samples, means one negative sample per positive sample
optimizer = optim.Adam(cdae_model.parameters(), lr=0.001)


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
        loss_values: 1D tensor of loss values, already in -loss form

    Returns:
        1D CDF weights aligned with loss_values
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
    return w_base * w_u_per_sample * w_i_per_sample


def lookup_sparse_weights(u, i, prev_keys, prev_w, n_items):
    """Lookup previous-epoch fused weights by (u,i); miss -> 1.0."""
    if prev_keys is None or prev_keys.numel() == 0:
        return torch.ones(u.shape[0], dtype=torch.float32)

    keys = u.long() * n_items + i.long()
    idx = torch.searchsorted(prev_keys, keys)
    idx_clamped = idx.clamp(max=prev_keys.numel() - 1)
    matched = (idx < prev_keys.numel()) & (prev_keys[idx_clamped] == keys)
    out = torch.ones(keys.shape[0], dtype=prev_w.dtype)
    out[matched] = prev_w[idx_clamped[matched]]
    return out


def build_sparse_weight_table(u, i, w, n_items):
    """Deduplicate (u,i) keeping last write (dense assignment semantics)."""
    if u.numel() == 0:
        empty_k = torch.empty(0, dtype=torch.long)
        empty_w = torch.empty(0, dtype=w.dtype)
        return empty_k, empty_w
    keys = u.long() * n_items + i.long()
    sorted_keys, sort_idx = torch.sort(keys)
    sorted_w = w[sort_idx]
    keep = torch.empty(sorted_keys.shape[0], dtype=torch.bool)
    keep[-1] = True
    if sorted_keys.numel() > 1:
        keep[:-1] = sorted_keys[:-1] != sorted_keys[1:]
    return sorted_keys[keep], sorted_w[keep]

########################### Eval #####################################
patience_count, best_hr = 0, 0  # Early stopping counter, best HR
best_recall = 0.0  # Best Recall
best_epoch = None  # The epoch corresponding to the best model
best_model_state = None  # Save the best model's state_dict in memory


def eval(model, valid_loader, valid_data_pos, train_mat, epoch):
    global best_recall, patience_count, best_epoch, best_model_state

    top_k = args.top_k
    model.eval()
    # model prediction can be more efficient instead of looping through each user, do it by batch
    predictedIndices_all = torch.empty(
        users_num, top_k[-1], dtype=torch.long
    )  # predictions
    GroundTruth = list(valid_data_pos.values())  # ground truth is exact item indices
    for user_valid, data_value_valid in valid_loader:
        with torch.no_grad():
            user_valid = user_valid.cuda()
            prediction_input_from_train = torch.tensor(
                train_mat[user_valid.cpu()]
            ).cuda()
            prediction = model(
                user_valid, prediction_input_from_train
            )  # prediction of the batch from train matrix
            valid_data_mask = (
                train_mat[user_valid.cpu()] * -9999
            )  # depends on the size of data

            prediction = prediction + torch.tensor(valid_data_mask).float().cuda()
            _, indices = torch.topk(prediction, top_k[-1])
            predictedIndices_all[user_valid.cpu()] = indices.cpu()

    predictedIndices = predictedIndices_all[list(valid_data_pos.keys())]
    precision, recall, NDCG, MRR = evaluate.compute_acc(
        GroundTruth, predictedIndices, top_k
    )

    print("################### EVAL ######################")
    print(f"Recall:{recall} NDCG: {NDCG}")

    if recall[0] > best_recall:
        best_recall = recall[0]
        best_epoch = epoch
        patience_count = 0
        # Save the best model's state_dict in memory to avoid frequent disk I/O
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"New best model found! Recall: {best_recall:.4f} at epoch {epoch}")
    else:
        patience_count += 1


########################### Test #####################################
def test(model, test_data_pos, train_mat, valid_mat):
    top_k = args.top_k
    model.eval()
    predictedIndices = []  # predictions
    GroundTruth = list(test_data_pos.values())

    for users in toolz.partition_all(
        1000, list(test_data_pos.keys())
    ):  # looping through users in test set
        user_id = torch.tensor(list(users)).cuda()
        data_value_test = torch.tensor(train_mat[list(users)]).cuda()
        predictions = model(user_id, data_value_test)  # model prediction for given data
        test_data_mask = (train_mat_dense[list(users)] + valid_mat[list(users)]) * -9999

        predictions = predictions + torch.tensor(test_data_mask).float().cuda()
        _, indices = torch.topk(
            predictions, top_k[-1]
        )  # returns sorted index based on highest probability
        indices = indices.cpu().numpy().tolist()
        predictedIndices += indices  # a list of top 100 predicted indices

    precision, recall, NDCG, MRR = evaluate.compute_acc(
        GroundTruth, predictedIndices, top_k
    )
    print("################### TEST ######################")
    print("Recall {:.4f}-{:.4f}".format(recall[0], recall[1]))
    print("NDCG {:.4f}-{:.4f}".format(NDCG[0], NDCG[1]))


########################### Training #####################################
startTime = time.time()
top_k = args.top_k
# Sparse previous-epoch fused weights: sorted keys (u * items_num + i) -> w
prev_keys = None
prev_w = None

for epoch in range(1000):
    cdae_model.train()
    train_loss = 0

    user_epoch_loss_sum = torch.zeros(users_num, dtype=torch.float)
    item_epoch_loss_sum = torch.zeros(items_num, dtype=torch.float)
    train_user_interaction_cnts = torch.zeros(users_num, dtype=torch.float)
    train_item_interaction_cnts = torch.zeros(items_num, dtype=torch.float)
    all_u = []
    all_i = []
    all_bce_losses = []

    for user, pos_interation_mat in train_loader:
        user = user.cuda()
        pos_interation_mat = pos_interation_mat.cuda()
        prediction = cdae_model(user, pos_interation_mat)
        # negative sampling
        with torch.no_grad():
            num_ns_per_user = pos_interation_mat.sum(1) * num_ns
            negative_samples = []
            idxs = []
            for idx in range(pos_interation_mat.size(0)):
                neg_item_list = torch.randint(
                    0, items_num, (int(num_ns_per_user[idx].item()),)
                )
                negative_samples.append(neg_item_list)
                idxs.extend([idx] * int(num_ns_per_user[idx].item()))

        # Create a mask to filter out valid samples; 1 indicates a valid sample, 0 indicates an invalid sample
        # This is equivalent to including negative samples in addition to the positives
        negative_samples = torch.cat(negative_samples, 0)
        idxs = torch.LongTensor(idxs)
        pos_neg_interaction_mat_mask = pos_interation_mat.clone()
        pos_neg_interaction_mat_mask[idxs, negative_samples] = 1

        # Get the labels and predictions for positive and negative samples
        groundtruth = pos_interation_mat[pos_neg_interaction_mat_mask > 0.0]
        pred = prediction[pos_neg_interaction_mat_mask > 0.0]

        loss = BCE_loss(pred, groundtruth)

        # Global user/item indices for all positive and negative samples in this batch
        user_idx, item_idx = torch.where(pos_neg_interaction_mat_mask > 0)
        global_u = user[user_idx].cpu()
        global_i = item_idx.cpu()
        loss_cpu = loss.detach().cpu()

        user_epoch_loss_sum.index_add_(0, global_u, loss_cpu)
        item_epoch_loss_sum.index_add_(0, global_i, loss_cpu)
        all_u.append(global_u)
        all_i.append(global_i)
        all_bce_losses.append(loss_cpu)

        # Based on interaction_mat_mask.sum(1), we get the number of interactions for each user in the current batch
        train_user_interaction_cnts[user.cpu()] += pos_neg_interaction_mat_mask.sum(
            1
        ).cpu()
        # Based on interaction_mat_mask.sum(0), we get the number of interactions for each item in the current batch
        train_item_interaction_cnts += pos_neg_interaction_mat_mask.sum(0).cpu()

        with torch.no_grad():
            if epoch <= 1:
                batch_ui_factor = torch.ones_like(loss)
            else:
                batch_ui_factor = lookup_sparse_weights(
                    global_u, global_i, prev_keys, prev_w, items_num
                ).cuda()

        # calculating loss
        batch_loss = torch.mean(batch_ui_factor * loss)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss

    print("epoch: {}, loss:{}".format(epoch, train_loss))

    all_u_t = torch.cat(all_u)
    all_i_t = torch.cat(all_i)
    all_loss_t = torch.cat(all_bce_losses)

    user_avg_loss = user_epoch_loss_sum / train_user_interaction_cnts.clamp(min=1.0)
    item_avg_loss = item_epoch_loss_sum / train_item_interaction_cnts.clamp(min=1.0)

    # Calculate user and item weight factors (PLUS: floor_ratio + confidence)
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

    w_base = get_based_weight(-all_loss_t)
    w_sparse = fuse_weights(w_base, user_fact, item_fact, all_u_t, all_i_t)
    prev_keys, prev_w = build_sparse_weight_table(
        all_u_t, all_i_t, w_sparse, items_num
    )

    all_u.clear()
    all_i.clear()
    all_bce_losses.clear()

    if epoch >= args.epoch_eval and epoch % args.eval_interval == 0:
        # validation
        eval(cdae_model, valid_loader, valid_pos, train_mat_dense, epoch)

        if patience_count == args.patience:
            break

print("############################## Training End. ##############################")
print(f"Whole Cost time: {(time.time()-startTime) / 60:.2f} min")
print(f"args: {args}")


########################### Test #####################################
model_save_path = os.path.join(
    model_path,
    f"CDAE_{args.dataset}_ratio-{args.floor_ratio}_upper-{args.factor_upper}.pth",
)

if best_model_state is not None:
    torch.save(best_model_state, model_save_path)
    print(f"Best model found in epoch {best_epoch}")
    print(f"Best model saved to: {model_save_path}")

    cdae_model.load_state_dict(best_model_state)
    cdae_model.cuda()
    test(cdae_model, test_pos, train_mat_dense, valid_mat_dense)
else:
    print("[WARN] No best model found during training; skip save/test.")
