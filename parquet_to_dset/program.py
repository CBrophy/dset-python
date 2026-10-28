import os
import sys
from typing import List, Optional

from easymorph.drivers import CancellationToken, DriverFactory

from .parquet_import_driver import read_data_from_file


def import_parquet_export_dset(source_file: str, destination_file: str) -> None:
    print(f"Started {source_file}")
    token = CancellationToken()
    dataset = read_data_from_file(source_file, token)

    DriverFactory.get_export_driver().export_dataset(dataset, destination_file, token)

    print(f"Finished {source_file}")


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv

    if len(args) != 2:
        print("Incorrect number of arguments. Expected two arguments: source file path and destination file path")
        return 1

    source_file, destination_file = args
    if not os.path.isfile(source_file):
        print(f"File does not exist: {source_file}")
        return 1

    if os.path.exists(destination_file):
        print(f"Destination file will be overwritten: {destination_file}")

    import_parquet_export_dset(source_file, destination_file)
    return 0
