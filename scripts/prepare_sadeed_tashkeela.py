from os import path
from sys import exit

from datasets import load_dataset


DATASET_NAME = "Sadeed_Tashkeela"
DATASET_DIST_PATH = path.join(path.dirname(__file__), "../datasets", DATASET_NAME)

DATASET = dict(
	{
		"train": [
			"hf://buckets/HUiDAALDHU/Sadeed_Tashkeela-bucket/train-00000-of-00003.parquet",
			"hf://buckets/HUiDAALDHU/Sadeed_Tashkeela-bucket/train-00001-of-00003.parquet",
			"hf://buckets/HUiDAALDHU/Sadeed_Tashkeela-bucket/train-00002-of-00003.parquet",
		],
		"test": [
			"hf://buckets/HUiDAALDHU/Sadeed_Tashkeela-bucket/test-00000-of-00001.parquet",
		],
	}
)

def main() -> None:
	ds = load_dataset("parquet", data_files=DATASET)

	ds["train"].to_csv(f"{DATASET_DIST_PATH}/train.csv"),
	ds["test"].to_csv(f"{DATASET_DIST_PATH}/test.csv")

if __name__ == "__main__":
	exit(main())
