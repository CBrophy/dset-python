"""Command-line utility that converts an EasyMorph .dset file into a Parquet file."""

from .parquet_export_driver import write_data, write_data_to_file

__all__ = ["write_data", "write_data_to_file"]
