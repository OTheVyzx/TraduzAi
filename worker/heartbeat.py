from __future__ import annotations

import threading

from worker.client import WorkerClient


class HeartbeatLoop:
    def __init__(self, client: WorkerClient, worker_id: str, interval_seconds: int):
        self.client = client
        self.worker_id = worker_id
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.client.heartbeat(self.worker_id)
            except Exception as exc:
                print(f"AVISO: heartbeat nao enviado: {exc}")
