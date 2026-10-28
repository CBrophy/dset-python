import argparse
import os
from typing import List, Optional

from easymorph.drivers import CancellationToken, DriverFactory

from .csv_import_driver import read_data_from_file


def import_csv_export_dset(
    source_file: str, destination_file: str, separator: str, infer_types: bool, encoding: str
) -> None:
    print(f"Started {source_file}")
    token = CancellationToken()
    dataset = read_data_from_file(source_file, separator, infer_types, encoding, token)

    DriverFactory.get_export_driver().export_dataset(dataset, destination_file, token)

    print(f"Finished {source_file}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="csv-to-dset", description="Converts a .csv file into an EasyMorph .dset file.")
    parser.add_argument("source", help="source .csv file path")
    parser.add_argument("destination", help="destination .dset file path")
    parser.add_argument("-s", "--separator", default=",", help="field separator (default: ',')")
    parser.add_argument("-e", "--encoding", default="utf-8-sig", help="source file encoding (default: UTF-8)")
    parser.add_argument(
        "--no-infer-types",
        dest="infer_types",
        action="store_false",
        help="import every value as text instead of detecting numbers, booleans and empty values",
    )
    args = parser.parse_args(argv)

    if args.separator == "\\t":
        args.separator = "\t"

    if not os.path.isfile(args.source):
        print(f"File does not exist: {args.source}")
        return 1

    if os.path.exists(args.destination):
        print(f"Destination file will be overwritten: {args.destination}")

    import_csv_export_dset(args.source, args.destination, args.separator, args.infer_types, args.encoding)
    return 0
