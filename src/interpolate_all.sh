#!/bin/bash

# create environment

# create alignment files from input csv & compute distances with RAxML
python src/0_create_raw_input/main.py --input-dir data/tiny_data --output-dir data

# format non-interpolated input
python src/2_dataset_formatting/no_interpolation/main.py \
  --data-dir data/no_interpolation \
  --species-info-dir data/species_info

# interpolate
python src/1_interpolation/main.py --records-file data/no_interpolation/Dactylorhiza/train.csv \
  --input-dir data/no_interpolation/raw_sorted_files/Dactylorhiza/alignments \
  --out data/imbalanced_interpolation/Dactylorhiza \
  --taxonomic-group 'Dactylorhiza' \
  --seqs-to-generate 200

python src/1_interpolation/main.py --records-file data/no_interpolation/Lomatium/train.csv \
  --input-dir data/no_interpolation/raw_sorted_files/Lomatium/alignments \
  --out data/imbalanced_interpolation/Lomatium \
  --taxonomic-group 'Lomatium' \
  --seqs-to-generate 80

python src/1_interpolation/main.py --records-file data/no_interpolation/Palaquium/train.csv \
  --input-dir data/no_interpolation/raw_sorted_files/Palaquium/alignments \
  --out data/imbalanced_interpolation/Palaquium \
  --taxonomic-group 'Palaquium' \
  --seqs-to-generate 30

python src/1_interpolation/main.py --records-file data/no_interpolation/Pterocarpus/train.csv \
  --input-dir data/no_interpolation/raw_sorted_files/Pterocarpus/alignments \
  --out data/imbalanced_interpolation/Pterocarpus \
  --taxonomic-group 'Pterocarpus' \
  --seqs-to-generate 40

# format interpolated input - imbalanced
python src/2_dataset_formatting/imbalanced_interpolation/main.py \
  --data-dir data/imbalanced_interpolation

# format interpolated input - balanced
python src/2_dataset_formatting/balanced_interpolation/main.py \
  --data-dir data/balanced_interpolation

# format test
pyton extract_test_samples.py \
  --base-dir data/no_interpolation
