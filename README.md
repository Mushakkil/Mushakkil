<h1 align="center">Mushakkil</h1>
<h3 align="center">An Arabic diacritization system</h3>

Mushakkil is an Arabic diacritization (tashkeel) system: given undiacritized
Arabic text, the goal is to restore the diacritics (harakat) that mark short
vowels and other pronunciation. The work in this repository covers the data
side of that objective — assembling large, diacritized Arabic corpora and
processing them into aligned training pairs, where each record holds a
diacritized sentence and the same sentence with its diacritics stripped
(`DIACRRITIC` / `NON_DIACRRITIC`).

#### Prerequisites

- **Python 3.13** (see Setup Enviroment above).
- **Both submodules initialized** (see Submodules above).
- **A JDK** — not just a JRE. Step 3 needs `javac` to compile
  `ShamelaIndexExporter.java`, and step 4 needs `java` on `PATH`
- **Disk space.** ~40 GB to hold the archive, the unpacked datasets, and the processed datasets

## Setup Enviroment

1. clone the project
2. install dependecies:

```bash
cd Mushakkil

# with uv
uv sync

# with bare-metal pip
pip install -r requirements.txt
```

### Submodules

The two data-preparation tools under `tools/dataset-prep/` are **git
submodules**, not files in this repository:

| Path                                   | Upstream                                                |
| -------------------------------------- | ------------------------------------------------------- |
| `tools/dataset-prep/shamela-extractor` | https://github.com/Mushakkil/shamela-extractor          |
| `tools/dataset-prep/dataset-preparer`  | https://github.com/Mushakkil/tashkeela_dataset_preparer |

***A plain `git clone` leaves both directories empty and the Shamela pipeline will
fail. Clone with them:***

```bash
git clone --recurse-submodules <repo-url>
```

Or, in an existing checkout:

```bash
git submodule update --init --recursive
```

## Data preparation

Two scripts under `scripts/` build the datasets. Both write into `datasets/`,
and `datasets/` and `datasets_archive/` are gitignored — the data is rebuilt
locally, never committed.

| Script                                | Source                                         | Output                                       |
| ------------------------------------- | ---------------------------------------------- | -------------------------------------------- |
| `scripts/prepare_shamela.py`          | The Shamela digital library database dump      | `datasets/shamela/*.csv`, one CSV per book   |
| `scripts/prepare_sadeed_tashkeela.py` | The `Sadeed_Tashkeela` dataset on Hugging Face | `datasets/Sadeed_Tashkeela/{train,test}.csv` |

## Train / Hypertune
you can start to train/hypertune the model from Mushakkil.ipynb

#### Furthrer Hypertuning
check our models builders at src/models.py.
There is three main models:
- Baseline model (Conv1D)
- Bidirectional/Single-Directional LSTM 
- Bidirectional/Single-Directional GRU 

##### Licenses

- `This system could be trained on Sadeed_Tashkeela (aka. Misraj/Sadeed_Tashkeela on HF) by Zeina Aldallal (and others) Which has been released under GNU General Public License version 2.0 (GPLv2), OSI-Approved Open Source (for research purposes only)`
- `Also it could be trained on Tashkeela corpus (T. Zerrouki, A. Balla, Tashkeela: Novel corpus of Arabic vocalized texts, data for auto-diacritization systems, Data in Brief (2017)) Which has been released under GNU General Public License version 2.0 (GPLv2), OSI-Approved Open Source`
- `Also it could be trained on Shamela Library dump which is a free-to-use project`