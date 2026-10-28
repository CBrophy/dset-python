import argparse
import os
from typing import List, Optional

from easymorph.drivers import DriverFactory, ImportConfig

from .parquet_export_driver import COMPRESSIONS, DEFAULT_COMPRESSION, write_data_to_file


def import_dset_export_parquet(config: ImportConfig, destination_file: str, compression: str) -> None:
    print(f"Started {config.file_name}")
    import_driver = DriverFactory.get_import_driver()
    datasets = import_driver.import_datasets(config)
    if len(datasets) == 0:
        print(f"Source file is empty, nothing to export: {config.file_name}")
        return

    write_data_to_file(datasets[0], destination_file, config.token, compression)

    print(f"Finished {config.file_name}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dset-to-parquet", description="Converts an EasyMorph .dset file into a Parquet file."
    )
    parser.add_argument("source", help="source .dset file path")
    parser.add_argument("destination", help="destination .parquet file path")
    parser.add_argument(
        "-c",
        "--compression",
        default=DEFAULT_COMPRESSION,
        choices=COMPRESSIONS,
        help=f"compression codec of the Parquet file (default: {DEFAULT_COMPRESSION})",
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.source):
        print(f"File does not exist: {args.source}")
        return 1

    if os.path.exists(args.destination):
        print(f"Destination file will be overwritten: {args.destination}")

    import_dset_export_parquet(ImportConfig(args.source), args.destination, args.compression)
    return 0
