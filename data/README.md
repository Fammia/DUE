# Data Inputs

Download the processed train/validation/test splits from [UDT](https://github.com/hyNTU/UDT/tree/cc2c8706105ee15b92a110bc07eb31e75f64d8d9/data) and place them in the directories below. Profile construction additionally requires `movies.dat` from [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/).

Expected layout:

```text
data/
  movielens/
    movielens.train.rating
    movielens.valid.rating
    movielens.test.negative
    movies.dat                 # additionally required by Profile-Pollution-Toolkit construction
  yelp/
    yelp.train.rating
    yelp.valid.rating
    yelp.test.negative
  amazon_book/
    amazon_book.train.rating
    amazon_book.valid.rating
    amazon_book.test.negative
```

`*.train.rating` and `*.valid.rating` contain tab-separated integer triples: `user_id item_id annotation`. Annotations identify clean (`1`), noisy (`0`), and injected (`2`) interactions for diagnostic grouping; DUE computes weights from training losses and support counts. Each `*.test.negative` line contains at least the integer pair `user_id item_id`; the LightGCN loader also accepts a third field. IDs are zero-based, and validation/test IDs must be within the user and item ranges inferred from the training split. The profile-pollution configuration uses 6,040 real MovieLens users and the supplied target item IDs.

Split sizes:

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| MovieLens-1M | 706,391 | 179,588 | 114,230 |
| Yelp | 1,672,520 | 209,453 | 163,308 |
| Amazon-book | 2,714,021 | 340,221 | 283,912 |

Check the inputs from the repository root:

```bash
python scripts/check_data.py --dataset all
python scripts/check_data.py --dataset all --paper-hash --profile-pollution
```

The first command checks split-file presence and format. `--paper-hash` also checks the SHA-256 values in `paper_split_sha256.txt`; `--profile-pollution` additionally checks `movies.dat` for profile construction.
