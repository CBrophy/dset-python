"""Reader for EasyMorph dataset files (.dset)."""

from .data import CellType, Column, Dataset, Nothing, ValueColumn
from .drivers import (
    CancellationToken,
    DriverFactory,
    DsetFormatError,
    ExportDriver,
    ImportConfig,
    ImportDriver,
    OperationCanceledError,
)


def read_dset(file_name: str) -> Dataset:
    """Convenience helper: imports a .dset file and returns its single dataset."""
    return ImportDriver().import_datasets(ImportConfig(file_name))[0]


def write_dset(dataset: Dataset, file_name: str) -> None:
    """Convenience helper: writes a dataset to a .dset file."""
    ExportDriver().export_dataset(dataset, file_name)


__all__ = [
    "CancellationToken",
    "CellType",
    "Column",
    "Dataset",
    "DriverFactory",
    "DsetFormatError",
    "ExportDriver",
    "ImportConfig",
    "ImportDriver",
    "Nothing",
    "OperationCanceledError",
    "ValueColumn",
    "read_dset",
    "write_dset",
]
