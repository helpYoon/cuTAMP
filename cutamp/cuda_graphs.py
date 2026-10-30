"""Capture and replay of the Adam loop's CUDA graphs (``--cuda_graphs``), after cuRobo's ``GraphExecutor``
(``curobo/_src/util/cuda_graph_util.py``). A graph bakes in every device address it touches, so
:class:`CapturedGraph` checks before each replay that none has moved."""

import time
import weakref
from functools import lru_cache
from typing import Any, Callable, Iterable, Optional

import torch
from curobo.kinematics import Kinematics


@lru_cache(maxsize=None)
def side_stream() -> torch.cuda.Stream:
    """The process-wide stream every warm-up and capture runs on."""
    return torch.cuda.Stream()


def graph_ptrs(tensors: Iterable[Optional[torch.Tensor]], kins: Iterable[Kinematics]) -> tuple:
    """A graph's guard tuple: each tensor's ``data_ptr()`` (0 for ``None``), then each Kinematics' batch,
    horizon and buffer pointers, which change when it is resized."""
    def ptr(t: Optional[torch.Tensor]) -> int:
        return 0 if t is None else t.data_ptr()

    out = [ptr(t) for t in tensors]
    for kin in kins:
        out += [kin._batch, kin._horizon, *map(ptr, kin._buffers.values())]
    return tuple(out)


class CapturedGraph:
    """A captured graph and its guard: ``outputs`` are the static tensors ``body`` returned in the capture,
    which every replay rewrites in place; ``ptrs`` reads the guard tuple (:func:`graph_ptrs`)."""

    def __init__(self, name: str, cuda_graph: torch.cuda.CUDAGraph, outputs: Any, body: Callable[[], Any],
                 ptrs: Callable[[], tuple], capture_s: float):
        self.name = name
        self.cuda_graph = cuda_graph
        self.outputs = outputs
        self.body = body
        self.ptrs = ptrs
        self.capture_s = capture_s
        self.captured_ptrs = ptrs()

    def replay(self) -> None:
        """Replay on the current stream once the guard holds: a moved address would touch freed memory."""
        now = self.ptrs()
        if now != self.captured_ptrs:
            moved = [i for i, (a, b) in enumerate(zip(now, self.captured_ptrs)) if a != b]
            raise AssertionError(f"{self.name}: guard entries {moved} of {len(self.captured_ptrs)} changed "
                                 f"since capture ({len(now)} entries now)")
        self.cuda_graph.replay()

    def release(self) -> None:
        """Drop the graph, its outputs, ``body`` and ``ptrs``. Its pool returns to the device at the next
        ``empty_cache()``, once no static output or graph-owned ``.grad`` still holds a block of it."""
        graph = weakref.ref(self.cuda_graph)
        self.cuda_graph = self.outputs = self.body = self.ptrs = None
        assert graph() is None, f"{self.name}: its CUDAGraph outlived release()"


def capture(name: str, body: Callable[[], Any], warmup: Callable[[], None], ptrs: Callable[[], tuple],
            warmups: int = 3) -> CapturedGraph:
    """Capture ``body`` as the CUDA graph ``name``, into its own private pool, after three ``warmup`` calls
    on :func:`side_stream` that fill lazy caches, compile kernels and size buffers (``warmup`` must leave
    no state the capture would bake in). ``"global"`` mode fails the capture on another thread's unsafe
    CUDA call. A failed capture leaves the side stream current, so this restores the caller's and raises;
    there is no eager fallback: the failure leaves torch's CUDA RNG unusable until a capture succeeds and
    the caching allocator in capture mode until the process exits.
    """
    side = side_stream()
    t0 = time.perf_counter()
    cur = torch.cuda.current_stream()
    side.wait_stream(cur)
    with torch.cuda.stream(side):
        for _ in range(warmups):
            warmup()
    cur.wait_stream(side)
    cuda_graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(cuda_graph, stream=side, capture_error_mode="global"):
            outputs = body()
    except Exception as e:
        torch.cuda.set_stream(cur)
        raise RuntimeError(f"CUDA graph capture of {name} failed; rerun without --cuda_graphs") from e
    torch.cuda.synchronize()
    return CapturedGraph(name, cuda_graph, outputs, body, ptrs, time.perf_counter() - t0)
