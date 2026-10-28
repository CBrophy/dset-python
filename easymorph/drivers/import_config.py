import threading
from typing import Optional


class OperationCanceledError(Exception):
    """Raised when an operation is canceled through a CancellationToken."""


class CancellationToken:
    """Thread-safe flag used to request cancellation of a running import or export."""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancellation_requested(self) -> bool:
        return self._event.is_set()

    def throw_if_cancellation_requested(self) -> None:
        if self._event.is_set():
            raise OperationCanceledError("The operation was canceled.")


class ImportConfig:
    """Settings to configure import."""

    def __init__(self, file_name: str, token: Optional[CancellationToken] = None):
        # File path for import
        self.file_name = file_name
        # Token to monitor cancellation requests
        self.token = token if token is not None else CancellationToken()
