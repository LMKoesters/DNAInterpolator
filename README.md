# DNAInterpolator

## Environment
To install all required packages for interpolation and training, download the environment.yml and requirements.txt files and run the following:

```bash
conda env create -f environment.yml
```

## Overview of files

### data
Contains tiny_data with the input data necessary to produce the alignment files needed for all analyses.
In addition, species_info covers all species information for the four datasets used in this study.

### src
#### 0_create_raw_input
This directory contains helper scripts to generate the files necessary to run the pipeline from data/tiny_data.
Run as:
```bash
python main.py --input-dir $YOUR_INPUT_DIR --output-dir $YOUR_OUTPUT_DIR
```

#### 1_interpolation
Here, the scripts needed for interpolating genetic data reside.
To run the interpolation pipeline, the main script should be called as:
```bash
python main.py --input-dir $ALIGNMENT_FOLDER_PATH --records-file $PATH_TO_RECORDS --taxonomic-group $NAME_OF_TAXONOMIC_GROUP
```

The pipeline offers additional arguments. Those are:
- --num-workers | -n: Workers for data loading (default: 4)
- --consensus-interpolation | -c: Refers to consensus sequence per species and locus to determine maximum number of nucleotides to change (overwrites interpolation radius, default: True)
- --interpolation-radius | -i: Takes distance to nearest neighbour and limits interpolation to a certain fraction of possible alterations (default: .4)
- --limit-interpolation | -w: Use radius to interpolate; otherwise use random number of alterations within all possible alterations and none
- --seqs-to-generate | -g: Max number of sequences to generate through interpolation (default: 50)
- --species-aware | -s: Only interpolate between sequences of the same species
- --patience | -p: A number that defines how many times the script should try to interpolate between sequences before moving on the next individual (default: 10)

#### 2_dataset_formatting
The original data and the data generated during interpolation needs to be formatted for training of the machine learning model. Again, formatting for each of the interpolation methods can be run by calling the main script.
```bash
python main.py
```

For formatting, users can set their usual data_dir as a default. Otherwise, --data-dir is a required argument. For formatting of non-interpolated data, providing the folder where dataframes with species labels per individual are stored, is another required argument.
Other options within the script are:

- --min-len | -i: Minimum length of sequences compared to median per gene_id/genus. Default: 0.5
- --max-len | -a: Maximum length of sequences compared to median per gene_id/genus. Default: None (i.e., not applying filtering on maximum length)
- --min-overlap | -o: Minimum overlap between sequence pairs. Default: 0.95

These options are used for filtering of DNA pairs.
The test datasets need to be set up separately. Within the script extract_test_samples.py, the base_dir needs to be defined first. Then, the test sets can be created by running:
```bash
pyton extract_test_samples.py
```

#### 3_training
To start training, the user can provide one of the three YAML files within the yamls directory.
Note that training needs to be started from within 3_training. Training can then be started by calling:
```bash
composer main.py --yaml-file $YAML_FILE_NAME
```

#### 4_eval
Evaluation closely resembles training. Therefore, the script again relies on the YAML file within the 4_eval/yamls directory. Evaluation can be started with:
```bash
composer main.py --yaml-file $YAML_FILE_NAME
```

