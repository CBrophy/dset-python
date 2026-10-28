from .export_driver import ExportDriver
from .import_driver import ImportDriver


class DriverFactory:
    """Factory to get drivers to work with datasets."""

    @staticmethod
    def get_import_driver() -> ImportDriver:
        return ImportDriver()

    @staticmethod
    def get_export_driver() -> ExportDriver:
        return ExportDriver()
