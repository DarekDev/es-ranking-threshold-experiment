# Dataset Directory

This directory contains the preprocessed RecSys 2015 data in session-item ranking format.

## Files Not Included in Git

Due to file size constraints, the following CSV files are **not included** in the repository:

- `train_data_recsys.csv` (~50MB) - 65,886 training session-item pairs
- `test_data_session_*.csv` (6 files, ~2MB total) - Test sessions for evaluation

## Generating the Dataset

To generate these files, follow these steps:

### 1. Download Raw Data

Download the RecSys Challenge 2015 dataset:
- **Official:** http://2015.recsyschallenge.com/challenge.html
- **Kaggle:** https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015

You need these files (~3GB total):
- `yoochoose-clicks.dat` (click events)
- `yoochoose-buys.dat` (purchase events)

### 2. Place Files

Create a directory structure:
```
EXPERIMENT_RECSYS/
├── SOFTWARE/
└── raw_data_download/    <- Create this folder
    ├── yoochoose-clicks.dat
    └── yoochoose-buys.dat
```

### 3. Run Preprocessing

```bash
cd ../SOFTWARE
python prepare_recsys_data.py
```

**Processing time:** ~5-10 minutes

### 4. Verify Output

You should see:
- `train_data_recsys.csv`: 65,886 rows × 10 columns
- `test_data_session_*.csv`: 6 files, ~40-45 rows each
- `dataset_metadata.json`: Statistics and configuration

## Dataset Metadata

The `dataset_metadata.json` file (included in git) contains:
- Dataset statistics (counts, distributions)
- Train/test split information
- Preprocessing parameters
- Random seeds for reproducibility

## Dataset Schema

Each CSV file has these columns:

| Column                        | Type   | Description                                    |
|-------------------------------|--------|------------------------------------------------|
| `session_primary_category`    | string | Most common category in session                |
| `session_browsing_pattern`    | string | low/medium/high click frequency                |
| `session_engagement_level`    | string | short/medium/long session duration             |
| `item_session_id`             | int    | Session identifier                             |
| `item_description`            | string | Composite text: category + features            |
| `item_purchased`              | bool   | Whether item was purchased in this session     |
| `item_features`               | string | Item-specific characteristics                  |
| `item_category`               | string | Item category code                             |
| `item_id`                     | int    | Item identifier                                |
| `target`                      | int    | Binary label: 1=purchased, 0=browsed only      |

## Dataset Statistics

- **Training size:** 65,886 session-item pairs
- **Test size:** 6 sessions × ~40 pairs = ~244 pairs total
- **Positive class:** 4.35% (purchased items)
- **Negative class:** 95.65% (browsed but not purchased)
- **Temporal split:** September 1, 2014
- **Feature dimensions:** 55,609 (8,000 TF-IDF + 47,609 one-hot)

## Data Quality

The preprocessing pipeline ensures:
- ✅ Zero train/test leakage (temporal split)
- ✅ Consistent categorical types (all strings)
- ✅ No missing values
- ✅ Balanced session representation
- ✅ Reproducible with fixed random seed (42)

## Troubleshooting

**Issue:** "File not found" error

**Solution:** Make sure you've downloaded the raw data and placed it in `../../raw_data_download/`

**Issue:** "Memory error" during preprocessing

**Solution:** The script uses parallel processing. You can disable it by setting `USE_PARALLEL = False` in `prepare_recsys_data.py`

**Issue:** Different results than paper

**Solution:** Check `dataset_metadata.json` matches the expected configuration. The random seed is fixed (42), so results should be exactly reproducible.

## Citation

Data source:
```bibtex
@inproceedings{ben2015recsys,
  title={RecSys Challenge 2015 and the YOOCHOOSE dataset},
  author={Ben-Shimon, D. and Tsikinovsky, A. and Friedmann, M. and Shapira, B. and Rokach, L. and Hoerle, J.},
  booktitle={Proceedings of the 9th ACM Conference on Recommender Systems},
  pages={357--358},
  year={2015}
}
```
