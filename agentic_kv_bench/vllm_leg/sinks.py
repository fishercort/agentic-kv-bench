"""Sink adapters: where the agent's metadata-only records go. The records are asserted
metadata-only before they reach a sink, so a sink never handles content.

- MemorySink: a bounded in-RAM sample + exact counts by kind (tests, CLI, short runs).
- JsonlSink: streams records to a JSON-lines file with no RAM growth (real deployment).

A production sink (Prometheus exporter for Grafana, ClickHouse, ...) is just another
implementation of this interface."""

from typing import Protocol


class Sink(Protocol):
    def emit(self, record: dict) -> None:
        ...

    def close(self) -> None:
        ...


class MemorySink:
    """Bounded in-RAM record sample + exact per-kind counts. cap bounds memory on a long run;
    counts stay exact regardless (the aggregate numbers never depend on the sample)."""

    def __init__(self, cap=None):
        self.cap = cap
        self.records = []
        self.counts = {}

    def emit(self, record):
        k = record.get("kind")
        self.counts[k] = self.counts.get(k, 0) + 1
        if self.cap is None or len(self.records) < self.cap:
            self.records.append(record)

    def close(self):
        pass


class JsonlSink:
    """Stream records to a JSON-lines file — the real-deployment sink (no RAM growth). Keeps
    exact per-kind counts for the run summary; holds no record sample."""

    def __init__(self, path):
        import json
        self._dumps = json.dumps
        self._f = open(path, "w")
        self.counts = {}
        self.records = []  # always empty; content streams to disk

    def emit(self, record):
        k = record.get("kind")
        self.counts[k] = self.counts.get(k, 0) + 1
        self._f.write(self._dumps(record) + "\n")

    def close(self):
        self._f.close()
