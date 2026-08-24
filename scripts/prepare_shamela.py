import sys
import os
from pathlib import Path
import urllib3

DATASET_DUMP_URL = "https://dev.shamela.ws/downloads/shamela-database-1448.zip"
ARCHIVE_DIST = (
	Path(__file__).resolve().parent
	/ ".."
	/ "datasets_archive"
	/ "shamela-database-1448.zip"
)

shamela_extractor_relative = "tools/dataset-prep/shamela-extractor"
shamela_extractor = Path(__file__).resolve().parent / ".." / shamela_extractor_relative

dataset_preparer_relative = "tools/dataset-prep/dataset-preparer"
dataset_preparer = Path(__file__).resolve().parent / ".." / dataset_preparer_relative

http = urllib3.PoolManager()


def main() -> None:
	return


def download_archive() -> None:
	"""
	Download the archive as chucks
	"""
	ARCHIVE_DIST.parent.mkdir(parents=True, exist_ok=True)

	with http.request("GET", DATASET_DUMP_URL, preload_content=False) as response:
		with open(ARCHIVE_DIST, "wb") as f:
			while True:
				chunk = response.read(1024 * 1024)	# 1 MB
				if not chunk:
					break
				f.write(chunk)


def unpack_archive() -> None:
	return


def build_jar() -> None:
	return


def extract_indices() -> None:
	return


def convert_into_text() -> None:
	return


def resolve_path(path) -> Path:
	return


if __name__ == "__main__":
	continue
