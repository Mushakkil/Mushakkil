# Handoff — Shamela data-prep pipeline

Time-sensitive status for whoever picks this up next. Durable setup and usage
instructions live in `README.md`; this file is about what state the code is in
right now.

Last updated: 2026-08-24

## What just changed

`scripts/prepare_shamela.py` was written but never run. Three reported bugs
were reproduced and fixed, and two further faults surfaced while reproducing
them.

### The three reported bugs

1. **`cleanup()` always crashed.** It called `os.rmdir()` on
   `datasets_archive/Shamela`, which is never empty — the extractor always
   leaves `exported_indices/` and `logs/` in it. Every otherwise-successful run
   ended in `OSError: [Errno 39] Directory not empty`. Now uses
   `shutil.rmtree()`, and no-ops when the directory is absent.

2. **`unpack_archive()` overwrote existing files.** `extractall()` clobbered an
   already-extracted tree on every re-run. It now iterates the zip's entries
   and skips any file already on disk, reporting an extracted/skipped count. A
   partially-extracted tree gets completed rather than replaced.

3. **`download_archive()` re-downloaded over an existing archive.** It now
   returns early when the archive is already present and non-empty. This is the
   difference between a no-op and a fresh 13.3 GB download.

### Two more faults found while reproducing those

4. **`download_archive()` could not run at all on a fresh checkout.**
   `ARCHIVE_DIST.parent.mkdir(...)` created the *repository root* instead of
   `datasets_archive/`, so the subsequent `open()` raised `FileNotFoundError`.
   Since `datasets_archive/` is gitignored, this hit on the very first run on
   any clean clone. Fixed to create the archive directory itself.

5. **`unpack_archive()` extracted relative to the current working directory.**
   It used `Path("datasets_archive")`, so running the script from anywhere but
   the repo root unpacked the archive next to wherever you stood, while the
   next step kept looking under the repo. It now extracts to an absolute path.
   Its bare `except:` also swallowed every error — a corrupt archive printed a
   message and the pipeline marched on to step 3 — so errors now propagate.

`download_archive()` additionally streams to a `.zip.part` file and renames it
on completion, so an interrupted 13.3 GB download is not mistaken for a
complete archive on the next run, and raises on a non-200 response instead of
writing an HTML error page to disk as a "zip".

## Verified vs. not verified

**Verified.** An offline harness (`verify.py`, kept out of the repo — ask if
you want it) loads the script, re-points its path constants at a temp
directory, and stubs the HTTP layer. It passes **11/11**:

- fresh-checkout download creates the directory and writes the archive
- an existing archive is skipped, with zero HTTP requests issued
- a zero-byte leftover is re-downloaded rather than treated as complete
- a non-200 response raises and leaves no archive behind
- no `.part` file is left behind on success
- extraction lands under `datasets_archive/` regardless of the working directory
- a re-run skips existing files instead of overwriting them
- a partial extraction is completed while existing files are kept
- a corrupt archive raises `BadZipFile` instead of being swallowed
- `cleanup()` removes a populated tree and keeps the archive
- `cleanup()` no-ops when there is nothing to remove

**Smoke-tested against the live host.** `dev.shamela.ws` is reachable; a `HEAD`
reports the archive as 13,294,044,352 bytes, `Content-Type: application/zip`.
The real `download_archive()` code path was run with a real
`urllib3.PoolManager` against a small URL on that same host — the streaming
loop, the status check, and the `.part` rename all work against a genuine
response.

**NOT verified — this is the important part.** The full 13.3 GB archive was
never downloaded, so nothing downstream of it has ever been executed:

- `build_jar()` — never run; needs a JDK
- `extract_indices()` — never run; needs the JDK and the real Lucene database
- `convert_into_text()` — never run
- `build_dataset()` — never run

The pipeline has therefore never completed end to end. The next person with a
machine that has ~40 GB free and a JDK installed should do a full run and
report back. Expect to find further problems in steps 3–6: those functions were
not part of this fix and have had no execution at all.

One specific thing to check on that first real run: `extract_indices()` passes
`datasets_archive` as `--shamela-path`, and the extractor looks for
`<shamela-path>/database/store`. Whether the archive actually unpacks
`database/store` directly into `datasets_archive/` — rather than nested under a
top-level folder such as `shamela4/` — could not be confirmed without
downloading it. If step 4 reports "Shamela Lucene store not found", this is
why, and the fix is to point `--shamela-path` at the real unpacked root.

## Known, unfixed

These were found but deliberately left alone, as they fall outside the reported
bugs. None of them are fixed.

1. **`dataset-preparer/src/app.py:98` writes output to a CWD-relative path.**
   It resolves the output directory as `Path("datasets") / "shamela"`, and
   `build_dataset()` does not set a working directory for the subprocess. The
   final dataset therefore lands wherever the script was launched from. This is
   the same class of bug as fault 5 above, still live — in a submodule, so
   fixing it means a change upstream in `tashkeela_dataset_preparer`.
   **Workaround: always run `prepare_shamela.py` from the repository root.**

2. **`cleanup()` only removes `datasets_archive/Shamela/`.** The unpacked
   database tree and `datasets_archive/Shamela_txt/` are both left behind —
   many gigabytes. Its scope was kept as written rather than expanded to delete
   more; widening what a cleanup function deletes is a decision worth making
   deliberately.

3. **The Java classpath relies on `javac` expanding the glob.** `build_jar()`
   passes `tools/dataset-prep/shamela-extractor/lib/*` as an argument in a
   `subprocess.run()` list, so no shell expands it — it reaches `javac`
   literally and depends on `javac`'s own handling of a `*` classpath entry. It
   resolves on Linux, but it is untested here and is a likely first failure on
   another platform.

Two smaller notes:

- **`urllib3` is imported by `prepare_shamela.py` but not declared in
  `pyproject.toml`.** It resolves today only transitively (via `datasets` →
  `requests`). If that transitive dependency ever changes, the script breaks
  with an `ImportError`. Worth adding as an explicit dependency.
- **`prepare_shamela.py` takes no arguments at all** — no `argparse`. In
  particular it does not expose the extractor's `--test-single-book`, so there
  is no way to do a quick single-book run through the wrapper. See the README
  for how to invoke the extractor directly. Wiring the flag through would make
  the first end-to-end run much cheaper to attempt.
