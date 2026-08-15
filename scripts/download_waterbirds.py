from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path


URL = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
ARCHIVE_NAME = "waterbird_complete95_forest2water2.tar.gz"


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if destination != target and destination not in target.parents:
            raise RuntimeError(f"unsafe path in archive: {member.name}")
    archive.extractall(destination, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the standard Waterbirds dataset.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / "metadata.csv").is_file() or (
        data_dir / "waterbird_complete95_forest2water2" / "metadata.csv"
    ).is_file():
        print(f"Waterbirds already prepared under {data_dir}")
        return

    archive_path = data_dir / ARCHIVE_NAME
    if not archive_path.is_file():
        partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
        print(f"Downloading {URL} -> {archive_path}")
        with urllib.request.urlopen(URL) as response, partial_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial_path.replace(archive_path)
    print(f"Extracting {archive_path}")
    with tarfile.open(archive_path, "r:gz") as archive:
        safe_extract(archive, data_dir)
    if not args.keep_archive:
        archive_path.unlink()
    print(f"Waterbirds is ready under {data_dir}")


if __name__ == "__main__":
    main()

