import glob
import random
import json
import os

import tensorflow as tf


class Dataset():
    """
    !! Haven't been put into practice !!

    ! DO NOT USE _book_level_split_balanced or _book_level_split_unbalanced, with dataset name: sadeed_tashkeal
    
    For NLP developers, you have two options to load the dataset:
        1. create Dataset object, then call obj.load_dataset(), easy, effecint. However:
            - You might be limited with tokenizers that comes keras/tensorflow
            - Taking samples might be a bit complex since you have to encode tensores values

        2. create Dataset object, then call obj._get_split_cache(). This will retrun:
            a. dict with two lists of filenames, train and test split. files that in CSV format.
               These splits are either balanced @see _book_level_split_balanced or 
               unbalanced @see _book_level_split_unbalanced (parameter choice)

        3. Also, you can use either _book_level_split_balanced or _book_level_split_unbalanced to create splits

    **Params**
        name: @see DATASET_REGISTRY static variable
        test_frac: the size of test splits. Books count in case of unbalanced split.
                   Rows count in case of balanced split (Default: 10%)
        seed: to make reproducible outputs. * Used with _book_level_split_unbalanced 
              and load_dataset's shuffle_seed
    """

    DATASET_REGISTRY = {
        "sadeed_tashkeal": {
            "train": "datasets/Sadeed_Tashkeela/train.csv",
            "test":  "datasets/Sadeed_Tashkeela/test.csv",
        },
        "shamela": "datasets/shamela/"
    }

    def __init__(
        self, 
        name: str, 
        test_frac: float=0.10, 
        seed: int=132,
        **kwargs
    ):
        self.name = name
        self.test_frac = test_frac
        self.csv_kwargs = kwargs
        self.seed = seed
                
        self._validate_config()


    def _validate_config(self) -> None:
        if self.name not in self.DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset: {self.name!r}. "
                f"Available: {list(self.DATASET_REGISTRY)}"
            )

        if not 0 < self.test_frac < 1:
            raise ValueError("test_frac must be between 0 and 1.")


    def _get_split_cache(self, split_type, path=".cache/shamela_split.json", purge_cache=False):
        """
        Prevent re-computing for splits
        """
        def _save_split(train_files, test_files):
            prev_split = _load_split()
            prev_split.update({split_type: {"train": train_files, "test": test_files}})

            cache_dir = os.path.dirname(path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(prev_split, f, indent=1)

        def _load_split():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    if not f.readable():
                        return {}

                    d = json.load(f)

                    if not isinstance(d, dict):
                        return {}
                return d
            except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                print(f"Warning: could not load split cache ({e}), starting fresh")
                return {}

        if self.name == "sadeed_tashkeal":
            return self.DATASET_REGISTRY[self.name]

        loaded_split = {}
        if not purge_cache:
            loaded_split = _load_split()

        if split_type not in loaded_split:
            if split_type == "balanced":
                train_files, test_files = self._book_level_split_balanced()[:2]
                _save_split(train_files=train_files, test_files=test_files)
            else:
                train_files, test_files = self._book_level_split_unbalanced()
                _save_split(train_files=train_files, test_files=test_files)


        return _load_split()[split_type]


    def _book_level_split_balanced(self) -> tuple[list, list, int, int]:
        """
        balanced split by row counting

        @retrun tuple(
            train_files,    # list of files by split
            test_files,     # //////
            train_rows,     # number of rows per split
            test_rows       # //////
        )
        @TODO dataset sampling
        @TODO dataset random shuffle
        fix: fix performance issue (approximate row counts using sizs??) 
        """

        csv_dir = self.DATASET_REGISTRY[self.name]
        files = sorted(glob.glob(f"{csv_dir}/*.csv"))
        
        if not files:
            raise FileNotFoundError(f"No CSV files found under {csv_dir!r}")

        # Get row count per book
        book_sizes = []
        for f in files:
            n_rows = sum(1 for _ in open(f)) - 1
            book_sizes.append((f, n_rows))

        total_rows = sum(size for _, size in book_sizes)
        target_test_rows = total_rows * self.test_frac

        # Sort largest-first: placing big items first makes greedy balancing much more accurate
        book_sizes.sort(key=lambda x: -x[1])

        train_files, test_files = [], []
        train_rows, test_rows = 0, 0

        for f, n_rows in book_sizes:
            if test_rows < target_test_rows:
                test_files.append(f)
                test_rows += n_rows
            else:
                train_files.append(f)
                train_rows += n_rows

        return train_files, test_files, train_rows, test_rows


    def _book_level_split_unbalanced(self) -> tuple[list, list]:
        """
        # NOT recommended for production: splits by book *count*, not rows count
        
        unbalanced split by book count, with random shuffle
        
        @retrun tuple(
            train_files,    # list of files by train split
            test_files,     # list of files by test split
        )
        """
        csv_dir = self.DATASET_REGISTRY[self.name]
        files = sorted(glob.glob(f"{csv_dir}/*.csv"))
        
        if not files:
            raise FileNotFoundError(f"No CSV files found under {csv_dir!r}")

        rng = random.Random(self.seed)
        rng.shuffle(files)

        n_test = int(len(files) * self.test_frac)
        test_files = files[:n_test]
        train_files = files[n_test:]
        
        return train_files, test_files

    def get_files_split(self, split: str, split_type: str = "balanced"):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
 
        if self.name == "sadeed_tashkeal":
            files = self.DATASET_REGISTRY[self.name][split]
 
        elif self.name == "shamela":
            if split_type == "balanced":
                train_files, test_files = self._get_split_cache("balanced")
            elif split_type == "unbalanced":
                train_files, test_files = self._get_split_cache("unbalanced")
            else:
                raise ValueError(f"Unknown split_type: {split_type!r}")
            files = train_files if split == "train" else test_files
 
        else:
            raise ValueError(f"No loading logic defined for dataset: {self.name!r}")

        return files

    def load_dataset(
        self, 
        split: str, 
        batch_size: int=32, 
        shuffle: bool=False,
        split_type: str = "balanced"
        ) -> tf.data.Dataset:
        """
        Load a tf.data.Dataset for the given split ("train" or "test").

        **params**
            split_type: "balanced" or "unbalanced" (default:"balanced") @see self._book_level_split_unbalanced and 
            self._book_level_split_balanced

            shuffle: shuffle the dataset (default: False)
 
        * split_type is ignored when dataset is sadeed_tashkeal
        """
        files = self.get_files_split(split, split_type)

        if not files:
            raise ValueError("There were no avalable datasets")

        ds = tf.data.experimental.make_csv_dataset(
            files,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_seed=self.seed,
            num_parallel_reads=tf.data.AUTOTUNE,
            **self.csv_kwargs,
        )

        # prefetch elements from the input dataset ahead of the time to
        # address performane issues with long loading time, even longer training.
        return ds.prefetch(tf.data.AUTOTUNE)
