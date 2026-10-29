"""Capture and replay of the Adam loop's CUDA graphs (``--cuda_graphs``).

A graph bakes in every device address its kernels touch, so it is only safe
while its inputs, its outputs and the hidden buffers it uses stay where they
were at capture. :class:`CapturedGraph` guards that before every replay, and
:func:`capture` follows cuRobo's ``GraphExecutor`` recipe
(``curobo/_src/util/cuda_graph_util.py``): warm-ups on a side stream, then
the capture on it, into the graph's own private memory pool.
"""

import time
import weakref
from functools import lru_cache
from typing import Any, Callable, Iterable, Optional

import torch
from curobo.kinematics import Kinematics


@lru_cache(maxsize=None)
def side_stream() -> torch.cuda.Stream:
    """The one stream every warm-up and capture runs on, created on first use
    and kept for the process."""
    return torch.cuda.Stream()


def graph_ptrs(tensors: Iterable[Optional[torch.Tensor]], kins: Iterable[Kinematics]) -> tuple:
    """A graph's guard tuple: the ``data_ptr()`` of each of ``tensors`` (0 for
    ``None``), then for each of ``kins`` its batch, its horizon and the
    ``data_ptr()`` of each of its ``_buffers`` (0 for ``None``). A resized
    Kinematics reallocates its buffers, so the tuple changes."""
    def ptr(t: Optional[torch.Tensor]) -> int:
        return 0 if t is None else t.data_ptr()

    out = [ptr(t) for t in tensors]
    for kin in kins:
        out += [kin._batch, kin._horizon, *map(ptr, kin._buffers.values())]
    return tuple(out)


class CapturedGraph:
    """A captured ``torch.cuda.CUDAGraph``, ``cuda_graph``, named ``name``.

    ``outputs`` is what ``body`` returned in the capture: static tensors that
    every replay rewrites in place. ``ptrs`` reads the guard tuple
    (:func:`graph_ptrs`), recorded right after the capture. ``capture_s`` is the
    wall time of the warm-ups and the capture.
    """

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
        """Replay the graph on the current stream, once the guard holds: an
        address that moved since capture would make the replay read and write
        freed memory, so a difference raises ``AssertionError`` first."""
        now = self.ptrs()
        if now != self.captured_ptrs:
            moved = [i for i, (a, b) in enumerate(zip(now, self.captured_ptrs)) if a != b]
            raise AssertionError(
                f"{self.name}: guard entries {moved} of {len(self.captured_ptrs)} changed since capture "
                f"({len(now)} entries now)"
            )
        self.cuda_graph.replay()

    def release(self) -> None:
        """Drop the graph, its outputs, ``body`` and ``ptrs``, and assert the
        ``torch.cuda.CUDAGraph`` is gone. Its private pool goes back to the
        device at the next ``empty_cache()``, once no tensor uses a block of
        it: a caller still holding a static output or a graph-owned ``.grad``
        keeps that block's segment."""
        graph = weakref.ref(self.cuda_graph)
        self.cuda_graph = self.outputs = self.body = self.ptrs = None
        assert graph() is None, f"{self.name}: its CUDAGraph outlived release()"


def capture(name: str, body: Callable[[], Any], warmup: Callable[[], None], ptrs: Callable[[], tuple],
            warmups: int = 3) -> CapturedGraph:
    """Capture ``body`` as the CUDA graph ``name`` (design 4.3, 4.5).

    ``warmups`` calls of ``warmup`` run first on :func:`side_stream`. They fill
    the lazy caches, compile kernels and size the buffers ``body`` uses, so the
    capture allocates and syncs nothing it shouldn't; ``warmup`` must leave no
    state a capture would bake in. The capture runs on the same stream in
    ``"global"`` mode, so another thread's unsafe CUDA call fails it loudly.
    Torch gives the graph its own private pool.

    A failed capture leaves the side stream current (the stream context's exit
    never runs once ``capture_end`` raises), so this sets the caller's stream
    back, then raises a ``RuntimeError`` naming ``--cuda_graphs``: there is no
    switch to eager, since a failed capture also leaves torch's CUDA RNG
    unusable until a later capture succeeds (design 4.8). A warm-up's error
    propagates as it is.
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
