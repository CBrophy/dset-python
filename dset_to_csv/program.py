import os
import sys
from typing import List, Optional

from easymorph.drivers import DriverFactory, ImportConfig

from .csv_export_driver import write_data_to_file


def import_dset_export_csv(config: ImportConfig, destination_file: str) -> None:
    print(f"Started {config.file_name}")
    import_driver = DriverFactory.get_import_driver()
    datasets = import_driver.import_datasets(config)
    if len(datasets) == 0:
        print(f"Source file is empty, nothing to export: {config.file_name}")
        return

    write_data_to_file(datasets[0], destination_file, ",", config.token)

    print(f"Finished {config.file_name}")


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

    import_dset_export_csv(ImportConfig(source_file), destination_file)
    return 0
