import os
import sys
from pathlib import Path
import zipfile
import urllib3
import subprocess

DATASET_DUMP_URL = "https://dev.shamela.ws/downloads/shamela-database-1448.zip"
BASE_DIR = Path(__file__).resolve().parent.parent

ARCHIVE_DIST = BASE_DIR / "datasets_archive"
DATASET_DIST = BASE_DIR / "datasets"
ARCHIVE_FILE = ARCHIVE_DIST / "shamela-database-1448.zip"

shamela_extractor_relative = "tools/dataset-prep/shamela-extractor"
shamela_extractor = BASE_DIR / shamela_extractor_relative

dataset_preparer_relative = "tools/dataset-prep/dataset-preparer"
dataset_preparer = BASE_DIR / dataset_preparer_relative

http = urllib3.PoolManager()

curr_python = sys.executable


def main() -> None:
	print("Starting dataset preparation...")

	print("\n[1/6] Downloading dataset archive...")
	download_archive()

	print("\n[2/6] Unpacking dataset archive...")
	unpack_archive()

	print("\n[3/6] Building Java index exporter...")
	build_jar()

	print("\n[4/6] Extracting indices...")
	extract_indices()

	print("\n[5/6] Converting CSV files to text...")

	convert_into_text()
	print("\n[6/6] building the dataset from text files...")
	build_dataset()

	print("\nCleaning up...")
	cleanup()

	print("\nDataset preparation finished successfully.")


def download_archive() -> None:
	"""
	Download the archive as chucks
	"""
	ARCHIVE_DIST.parent.mkdir(parents=True, exist_ok=True)

	with http.request("GET", DATASET_DUMP_URL, preload_content=False) as response:
		with open(ARCHIVE_FILE, "wb") as f:
			while True:
				chunk = response.read(1024 * 1024)	# 1 MB
				if not chunk:
					break
				f.write(chunk)


def unpack_archive() -> None:
	OUTPUT_DIR = Path("datasets_archive")
	OUTPUT_DIR.mkdir(exist_ok=True)

	archives = list(ARCHIVE_DIST.glob("*.zip"))

	if not archives:
		raise FileNotFoundError(f"No .zip archive found in {ARCHIVE_DIST}")

	dist = os.path.join(OUTPUT_DIR, "Shamela_dataset")

	try:
		with zipfile.ZipFile(archives[0]) as zipFile:
			zipFile.extractall(OUTPUT_DIR)
	except:
		print("Error occured while extraction of the dataset archive")

	return dist


def build_jar() -> None:
	subprocess.run(
		[
			"javac",
			"-cp",
			shamela_extractor / "lib/*",
			shamela_extractor / "java/ShamelaIndexExporter.java",
		],
		check=True,
	)


def extract_indices() -> None:
	subprocess.run(
		[
			curr_python,
			shamela_extractor / "extract_indices.py",
			"--shamela-path",
			ARCHIVE_DIST,
			"--output-path",
			ARCHIVE_DIST / "Shamela",
		],
		check=True,
	)


def convert_into_text() -> None:
	subprocess.run(
		[
			curr_python,
			shamela_extractor / "csv_to_txt.py",
			"-i",
			ARCHIVE_DIST / "Shamela/exported_indices/book_data" ,
			"-o",
			ARCHIVE_DIST / "Shamela_txt",
		],
		check=True,
	)


def build_dataset() -> None:
	subprocess.run(
		[
			curr_python,
			dataset_preparer / "src" / "app.py",
			"--target-dir",
			ARCHIVE_DIST / "Shamela_txt",
		],
		check=True,
	)


def cleanup() -> None:
	os.rmdir(ARCHIVE_DIST / "Shamela")
	return

if __name__ == "__main__":
	main()
