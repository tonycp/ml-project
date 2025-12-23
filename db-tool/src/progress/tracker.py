from dataclasses import dataclass
from pathlib import Path

import threading
import time


@dataclass
class TrackerState:
    percent: int = 0
    status: str = "waiting"  # waiting, loading, done, error
    start_time: float = 0
    error: str = ""


class ProgressTracker:
    def __init__(self, service_name: str, sql_file: Path):
        self.name = service_name
        self.sql_file = sql_file
        self.state = TrackerState()
        self._lock = threading.Lock()
        self._file_size_mb = sql_file.stat().st_size / (1024 * 1024)

    def update(self, percent: int, status: str, error: str = ""):
        with self._lock:
            self.state.percent = min(100, max(0, percent))
            self.state.status = status
            self.state.error = error
            if status == "loading" and not self.state.start_time:
                self.state.start_time = time.time()

    def get_state(self) -> TrackerState:
        with self._lock:
            return TrackerState(**self.state.__dict__)
