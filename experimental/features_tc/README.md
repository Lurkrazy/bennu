# features_tc Folder Description

## Main Functionality

This directory is used to load all tuning records from the TVM meta-schedule database (e.g., JSONDatabase) and extract the features of each record, ultimately saving them as a CSV file. Each record corresponds to one row of feature vectors.

## Key Workflow

1. Load the database (e.g., `data/ms/layer_mini_0/database_tuning_record.json`).
2. Iterate through all tuning records and restore the schedule.
3. Use `PerStoreFeature` to extract the features of each record.
4. Save all features as a CSV file, with each row corresponding to one record.

## Feature Explanation

- Feature shape is like (3, 164): 3 indicates that the current workload/store has 3 storage points (stages), and each point outputs 164-dimensional features.
- These features are concatenated from multiple groups (e.g., operators, loops, memory, etc.). See `ref/feature_extractor/per_store_feature.cc` for details.

## Reference Implementations

- [`ref/db_to_features.py`](ref/db_to_features.py): Demonstrates how to restore records from the database and extract features.
- [`load_from_db_to_features.py`](load_from_db_to_features.py): Implements the process of batch extraction and saving features to CSV.

## Typical Output

- `features.csv`: Each row is a feature vector of a tuning record, facilitating subsequent analysis or use in machine learning.
