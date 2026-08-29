!! Haven't been put into practice !!

! DO NOT USE _book_level_split_balanced or _book_level_split_unbalanced, with dataset name: sadeed_tashkeal !

## For NLP developers

you have two options to load the dataset:
1. create Dataset object, then call obj.load_dataset(), easy, effecint. However:
    - You might be limited with tokenizers that comes keras/tensorflow
    - Taking samples might be a bit complex since you have to encode tensores values

2. create Dataset object, then call obj._get_split_cache(). This will retrun:
    a. dict with two lists of filenames, train and test split. files that in CSV format.
       These splits are either balanced @see _book_level_split_balanced or
       unbalanced @see _book_level_split_unbalanced (parameter choice)
       
3. Also, you can use either _book_level_split_balanced or _book_level_split_unbalanced to create splits

## Params
- name: @see DATASET_REGISTRY static variable
- test_frac: the size of test splits. Books count in case of unbalanced split. Rows count in case of balanced split (Default: 10%)
- seed: to make reproducible outputs. * Used with _book_level_split_unbalanced and load_dataset's shuffle_seed
