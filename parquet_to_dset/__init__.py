"""Command-line utility that converts a Parquet file into an EasyMorph .dset file."""

from .parquet_import_driver import read_data, read_data_from_file

__all__ = ["read_data", "read_data_from_file"]
