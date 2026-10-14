import math
import toolz

import torch


def compute_acc(GroundTruth, predictedIndices, topN):
    precision = []
    recall = []
    NDCG = []
    MRR = []

    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        dcg += 1.0 / math.log2(j + 2)
                        if mrrFlag:
                            userMRR = 1.0 / (j + 1.0)
                            mrrFlag = False
                        userHit += 1

                    if idcgCount > 0:
                        idcg += 1.0 / math.log2(j + 2)
                        idcgCount = idcgCount - 1

                if idcg != 0:
                    ndcg += dcg / idcg

                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])
                sumForNdcg += ndcg
                sumForMRR += userMRR

        precision.append(sumForPrecision / len(predictedIndices))
        recall.append(sumForRecall / len(predictedIndices))
        NDCG.append(sumForNdcg / len(predictedIndices))
        MRR.append(sumForMRR / len(predictedIndices))

    return precision, recall, NDCG, MRR


def eval_all_rank(model, eval_pos, mask_mat, items_num, top_k, device, batch_users=1024):
    model.eval()
    predictedIndices = []
    GroundTruth = []
    users_in_eval = list(eval_pos.keys())

    with torch.no_grad():
        for users_batch in toolz.partition_all(batch_users, users_in_eval):
            users_batch = list(users_batch)
            GroundTruth.extend([eval_pos[u] for u in users_batch])
            users_t = torch.tensor(users_batch, dtype=torch.long, device=device)

            prediction = model.getUsersRating(users_t)
            # all-rank mask: hide observed interactions
            prediction = prediction + mask_mat[users_batch].to(device) * -9999
            _, indices = torch.topk(prediction, max(top_k))
            predictedIndices.extend(indices.cpu().numpy().tolist())

    precision, recall, NDCG, MRR = compute_acc(GroundTruth, predictedIndices, top_k)
    return precision, recall, NDCG, MRR
