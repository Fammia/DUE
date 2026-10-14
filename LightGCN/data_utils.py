import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as data


def load_all(dataset, data_path):
    train_rating = data_path + "{}.train.rating".format(dataset)
    valid_rating = data_path + "{}.valid.rating".format(dataset)
    test_negative = data_path + "{}.test.negative".format(dataset)

    ################# load training data #################
    train_data = pd.read_csv(
        train_rating,
        sep="\t",
        header=None,
        names=["user", "item", "noisy"],
        usecols=[0, 1, 2],
        dtype={0: np.int32, 1: np.int32, 2: np.int32},
    )
    users_num = train_data["user"].max() + 1
    items_num = train_data["item"].max() + 1
    print("user, item num")
    print(users_num, items_num)
    train_data = train_data.values.tolist()

    train_mat = sp.dok_matrix((users_num, items_num), dtype=np.float32)
    train_valid_mat = sp.dok_matrix((users_num, items_num), dtype=np.float32)
    train_features_list = []
    train_data_noisy = []
    train_user_pos = {}
    train_user_pos_set = {}
    for x in train_data:
        u, i, noisy = x[0], x[1], x[2]
        train_mat[u, i] = 1.0
        train_valid_mat[u, i] = 1.0
        train_features_list.append([u, i])
        train_data_noisy.append(noisy)
        if u in train_user_pos:
            train_user_pos[u].append(i)
            train_user_pos_set[u].add(i)
        else:
            train_user_pos[u] = [i]
            train_user_pos_set[u] = {i}

    ################# load validation data #################
    valid_data = pd.read_csv(
        valid_rating,
        sep="\t",
        header=None,
        names=["user", "item", "noisy"],
        usecols=[0, 1, 2],
        dtype={0: np.int32, 1: np.int32, 2: np.int32},
    )
    valid_data = valid_data.values.tolist()
    valid_features_list = []
    valid_pos = {}
    train_valid_pos = {}
    for u, pos_items in train_user_pos.items():
        train_valid_pos[u] = list(pos_items)

    for x in valid_data:
        u, i = x[0], x[1]
        train_valid_mat[u, i] = 1.0
        valid_features_list.append([u, i])
        if u in valid_pos:
            valid_pos[u].append(i)
        else:
            valid_pos[u] = [i]

        if u in train_valid_pos:
            train_valid_pos[u].append(i)
        else:
            train_valid_pos[u] = [i]

    ################# load testing data #################
    test_pos = {}
    has_noisy_flag = False
    with open(test_negative, "r") as fd:
        line = fd.readline()
        while line is not None and line != "":
            arr = line.strip().split("\t")
            if len(arr) < 2:
                line = fd.readline()
                continue

            u = int(arr[0])
            i = int(arr[1])
            if len(arr) >= 3:
                has_noisy_flag = True
                noisy = int(arr[2])
                if noisy != 0:
                    line = fd.readline()
                    continue
            # If test file has only two columns, it does not contain noisy flag.
            # In this case we keep the original repository behavior and cannot filter further.
            if u in test_pos:
                test_pos[u].append(i)
            else:
                test_pos[u] = [i]
            line = fd.readline()

    if not has_noisy_flag:
        print(
            "Warning: test.negative has no noisy field; cannot perform additional clean filtering."
        )

    return (
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
    )


class BPRData(data.Dataset):
    def __init__(self, features, items_num, train_user_pos_set):
        super(BPRData, self).__init__()
        self.features = features
        self.items_num = items_num
        self.train_user_pos_set = train_user_pos_set

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        u, i_pos = self.features[idx][0], self.features[idx][1]
        i_neg = np.random.randint(self.items_num)
        while i_neg in self.train_user_pos_set.get(u, set()):
            i_neg = np.random.randint(self.items_num)
        return u, i_pos, i_neg
