# Dataset Scripts

This directory contains scripts used to **download, prepare, convert, and export datasets** used by the project.

The scripts currently cover two main datasets:

* **Sadeed Tashkeela** — a diacritized Arabic text dataset hosted on Hugging Face.
* **Shamela** — the Shamela database archive, which is downloaded and processed into the project's dataset format.

---

## Directory Structure

```text
tools/
└── dataset-prep/
    ├── ...
    ├── dataset-preparer/
    └── shamela-extractor/
```

The generated datasets are stored under:

```text
datasets/
```

and downloaded archives are stored under:

```text
datasets_archive/
```

---

# 1. Sadeed Tashkeela

The Sadeed Tashkeela preparation script downloads the dataset from Hugging Face and converts the Parquet shards into CSV files.

## Dataset

The dataset is named:

```python
Sadeed_Tashkeela
```

The input consists of:

```text
train/
├── train-00000-of-00003.parquet
├── train-00001-of-00003.parquet
└── train-00002-of-00003.parquet

test/
└── test-00000-of-00001.parquet
```

The script loads these files using `datasets.load_dataset()` and exports them as:

```text
datasets/
└── Sadeed_Tashkeela/
    ├── train.csv
    └── test.csv
```

## Running

From the project root:

```bash
python <sadeed-tashkeela-script>.py
```

The script will:

1. Download/read the Parquet shards from Hugging Face.
2. Construct the `train` split.
3. Construct the `test` split.
4. Export both splits to CSV.

### Requirements

The script requires the Hugging Face `datasets` package:
---

# 2. Shamela

The Shamela preparation workflow starts from the official Shamela database archive:

```text
shamela-database-1448.zip
```

The archive is downloaded from:

```text
https://dev.shamela.ws/downloads/shamela-database-1448.zip
```

The downloaded archive is stored in:

```text
datasets_archive/
└── shamela-database-1448.zip
```

## Processing pipeline

The Shamela workflow consists of two stages:

```text
Shamela database
       │
       ▼
Download archive
       │
       ▼
datasets_archive/
       │
       ▼
Shamela extractor
       │
       ▼
Extracted text
       │
       ▼
Dataset preparer
       │
       ▼
Prepared CSV dataset
```

The scripts use the following tools:

### Shamela extractor

```text
tools/dataset-prep/shamela-extractor
```

This tool is responsible for extracting/transforming the downloaded Shamela database into usable text files.

### Dataset preparer

```text
tools/dataset-prep/dataset-preparer
```

This tool performs the subsequent corpus preparation, including preprocessing and sentence-level processing.

---

# Dataset Locations

The scripts intentionally keep downloaded archives separate from processed datasets.

```text
datasets_archive/
    └── shamela-database-1448.zip

datasets/
    ├── Sadeed_Tashkeela/
    │   ├── train.csv
    │   └── test.csv
    │
    └── ...
```

This separation makes it possible to:

* Re-run dataset preparation without downloading the archive again.
* Keep raw data separate from generated data.
* Remove generated datasets without losing the original archive.
* Reproduce the preprocessing pipeline.

---

# Dependencies

Depending on which script is being used, the dataset preparation tools may require:
- kagglehub
- pandas
- pyarabic
- regex
The project-specific tools under:

```text
tools/dataset-prep/shamela-extractor
tools/dataset-prep/dataset-preparer
```
---


# Adding a New Dataset

A new dataset preparation script should follow the same general pattern:

1. Define a unique dataset name.
2. Define the source dataset/archive.
3. Define the destination under `datasets/`.
4. Download or load the source data.
5. Apply the required preprocessing.
6. Export the processed dataset.
7. Document the dataset and reproduction steps in this README.

For example:

```python
DATASET_NAME = "My_Dataset"
DATASET_DIST_PATH = Path(__file__).resolve().parent / ".." / "datasets" / DATASET_NAME
```
