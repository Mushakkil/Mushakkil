<h1 align="center">Mushakkil</h1>
<h3 align="center">An Arabic diacritization system</h3>

Mushakkil is an Arabic diacritization (tashkeel) system: given undiacritized
Arabic text, the goal is to restore the diacritics (harakat) that mark short
vowels and other pronunciation. The work in this repository covers the data
side of that objective — assembling large, diacritized Arabic corpora and
processing them into aligned training pairs, where each record holds a
diacritized sentence and the same sentence with its diacritics stripped
(`DIACRRITIC` / `NON_DIACRRITIC`). These pairs are the supervision signal for
the diacritization model.

## Setup Enviroment
> [!CAUTION]
> you need uv to run this project

1. clone the project
2. install dependecies:
```bash
cd Mushakkil
uv sync
```

### Submodules

The two data-preparation tools under `tools/dataset-prep/` are **git
submodules**, not files in this repository:

| Path | Upstream |
| --- | --- |
| `tools/dataset-prep/shamela-extractor` | https://github.com/Mushakkil/shamela-extractor |
| `tools/dataset-prep/dataset-preparer` | https://github.com/Mushakkil/tashkeela_dataset_preparer |

A plain `git clone` leaves both directories empty and the Shamela pipeline will
fail. Clone with them:

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

| Script | Source | Output |
| --- | --- | --- |
| `scripts/prepare_shamela.py` | The Shamela digital library database dump | `datasets/shamela/*.csv`, one CSV per book |
| `scripts/prepare_sadeed_tashkeela.py` | The `Sadeed_Tashkeela` dataset on Hugging Face | `datasets/Sadeed_Tashkeela/{train,test}.csv` |

### Sadeed Tashkeela

`scripts/prepare_sadeed_tashkeela.py` loads the dataset's Parquet shards (three
train shards, one test shard) with `datasets.load_dataset()` and writes each
split to CSV:

```bash
uv run python scripts/prepare_sadeed_tashkeela.py
```

### Shamela

`scripts/prepare_shamela.py` turns the Shamela library dump into a plain-text
Arabic corpus and then into the project's CSV dataset format. It runs six
steps end to end:

1. Download `shamela-database-1448.zip` from `dev.shamela.ws` into `datasets_archive/`.
2. Unpack it into `datasets_archive/`.
3. Compile the Java Lucene index exporter (`javac`).
4. Extract the Lucene indices to CSV via that exporter.
5. Convert the per-book CSVs to plain `.txt` (the `body` column, one passage per block).
6. Run the dataset preparer over the text files to produce the final per-book CSVs.

It then removes the intermediate `datasets_archive/Shamela/` tree. The
downloaded archive is deliberately kept so the pipeline can be re-run without
downloading again.

#### Prerequisites

- **Python 3.13** and `uv` (see Setup Enviroment above).
- **Both submodules initialized** (see Submodules above).
- **A JDK** — not just a JRE. Step 3 needs `javac` to compile
  `ShamelaIndexExporter.java`, and step 4 needs `java` on `PATH`; the extractor
  checks for it and aborts if it is missing.
- **Disk space.** The archive is ~13.3 GB, and the extractor's own
  prerequisite check refuses to run when the output path has less than 20 GB
  free. Budget ~40 GB to hold the archive, the unpacked database, and the
  extracted CSV/text intermediates at once.
- **Network access** to `dev.shamela.ws`.

#### Running

Run from the repository root — step 6 writes its output to `datasets/shamela/`
relative to the current working directory:

```bash
uv run python scripts/prepare_shamela.py
```

The script takes no arguments. It is re-runnable: the download is skipped when
the archive is already present, unpacking skips files already on disk, and the
dataset preparer skips books whose CSV already exists.

#### Test run on a single book

`prepare_shamela.py` itself has no test mode, but the extractor it calls does.
To exercise the Java extraction path on one book instead of the whole library,
run the extractor directly after the archive is unpacked and the exporter is
compiled:

```bash
# compile the exporter (quote the classpath so the shell does not expand it)
javac -cp "tools/dataset-prep/shamela-extractor/lib/*" \
      tools/dataset-prep/shamela-extractor/java/ShamelaIndexExporter.java

# extract a single book
uv run python tools/dataset-prep/shamela-extractor/extract_indices.py \
      --shamela-path datasets_archive \
      --output-path datasets_archive/Shamela \
      --test-single-book
```

`--shamela-path` must point at a directory containing `database/store`, which
is where the extractor looks for the Lucene index.
