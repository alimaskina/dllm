from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import torch


@dataclass(slots=True)
class TimingRecord:
    name: str
    milliseconds: float
    layer: int | None = None
    block: int | None = None
    step: int | None = None
    stream: str = "main"


@dataclass(slots=True)
class SelectionRecord:
    block: int
    layer: int
    old_cache_len: int
    selected_k: int
    selector_queries: list[int]
    semantic: str
    index_checksum: int
    mean_score: float | None = None


@dataclass(slots=True)
class RunTrace:
    timings: list[TimingRecord] = field(default_factory=list)
    selections: list[SelectionRecord] = field(default_factory=list)
    counters: dict[str, int | float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add_counter(self, name: str, value: int | float) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def to_dict(self) -> dict:
        return {
            "timings": [asdict(x) for x in self.timings],
            "selections": [asdict(x) for x in self.selections],
            "counters": dict(self.counters),
            "notes": list(self.notes),
        }

    def dump(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


@dataclass(slots=True)
class _PendingCudaTiming:
    name: str
    start: torch.cuda.Event
    end: torch.cuda.Event
    layer: int | None
    block: int | None
    step: int | None
    stream: str


class RuntimeTimer:

    def __init__(self, trace: RunTrace, enabled: bool = False) -> None:
        self.trace = trace
        self.enabled = enabled
        self._pending: list[_PendingCudaTiming] = []

    @contextmanager
    def region(
        self,
        name: str,
        *,
        tensor: torch.Tensor | None = None,
        layer: int | None = None,
        block: int | None = None,
        step: int | None = None,
        stream_name: str = "main",
        stream: torch.cuda.Stream | None = None,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        use_cuda = tensor is not None and tensor.device.type == "cuda"
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            active = stream or torch.cuda.current_stream(tensor.device)
            start.record(active)
            try:
                yield
            finally:
                end.record(active)
                self._pending.append(
                    _PendingCudaTiming(
                        name, start, end, layer, block, step, stream_name
                    )
                )
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self.trace.timings.append(
                    TimingRecord(
                        name=name,
                        milliseconds=(time.perf_counter() - t0) * 1000.0,
                        layer=layer,
                        block=block,
                        step=step,
                        stream=stream_name,
                    )
                )

    def finalize(self) -> None:
        if not self._pending:
            return
        torch.cuda.synchronize()
        for item in self._pending:
            self.trace.timings.append(
                TimingRecord(
                    name=item.name,
                    milliseconds=float(item.start.elapsed_time(item.end)),
                    layer=item.layer,
                    block=item.block,
                    step=item.step,
                    stream=item.stream,
                )
            )
        self._pending.clear()
