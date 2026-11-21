"""Terminal state and exact first-order pullback of a T1 post-retract HOLD."""

from contextlib import contextmanager
from threading import Lock

import torch

_CHUNK = 5
_TICKS = 50


def _advance(model, state, hold, dynamics, ticks):
    for _ in range(ticks):
        state = model(state, hold, dynamics)
    return state


def _record(tensors, stream):
    for value in tensors:
        if value is not None and value.is_cuda:
            value.record_stream(stream)


class _Chunk:
    """All copies, replays and private graph allocations use one fixed CUDA stream."""

    def __init__(self, model, state, dynamics):
        # An outstanding backward can retain this chunk past predictor cache eviction.
        self.model = model
        self.lock = Lock()
        self.stream = torch.cuda.Stream(device=dynamics.device)
        self.forward_graph = torch.cuda.CUDAGraph()
        self.backward_graph = torch.cuda.CUDAGraph()
        with self.on_stream((*state, dynamics, *model.parameters(), *model.buffers())), torch.enable_grad():
            inputs = tuple(v.detach().contiguous().clone().requires_grad_(True) for v in state)
            context = dynamics.detach().contiguous().clone().requires_grad_(dynamics.requires_grad)
            self.inputs = (*inputs, context)
            self.hold = context.new_zeros(len(context), 5)

            def forward():
                return _advance(model, inputs, self.hold, context, _CHUNK)

            def backward(outputs, adjoints):
                active = [(o, g) for o, g in zip(outputs, adjoints) if o.requires_grad]
                xs = self.inputs if context.requires_grad else inputs
                return torch.autograd.grad(
                    tuple(o for o, _ in active), xs, tuple(g for _, g in active), allow_unused=True
                )

            trial = forward()
            self.adjoints = tuple(torch.zeros_like(v) for v in trial)
            backward(trial, self.adjoints)
            del trial

            self.forward_graph.capture_begin(capture_error_mode="thread_local")
            try:
                outputs = forward()
            finally:
                self.forward_graph.capture_end()
            self.backward_graph.capture_begin(capture_error_mode="thread_local")
            try:
                grads = backward(outputs, self.adjoints)
            finally:
                self.backward_graph.capture_end()
            self.outputs = tuple(x.detach() for x in outputs)
            self.grads = grads if context.requires_grad else (*grads, None)

    @contextmanager
    def on_stream(self, inputs):
        caller = torch.cuda.current_stream(self.stream.device)
        # A caller on another thread may hold an outstanding forward for this chunk.
        # Enqueue each entire transaction before another can overwrite static buffers.
        with self.lock:
            if caller == self.stream:
                yield caller
                return
            self.stream.wait_stream(caller)
            _record(inputs, self.stream)
            try:
                with torch.cuda.stream(self.stream):
                    yield caller
            finally:
                caller.wait_stream(self.stream)

    def advance(self, state):
        for dst, src in zip(self.inputs, state):
            dst.copy_(src)
        self.forward_graph.replay()

    def pullback(self, state, adjoints):
        # Recover the correct intermediate activations even after other outstanding forwards.
        self.advance(state)
        for dst, src in zip(self.adjoints, adjoints):
            if src is None:
                dst.zero_()
            else:
                dst.copy_(src)
        self.backward_graph.replay()


class _Hold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, chunk, *args):
        state, dynamics = args[:-1], args[-1]
        ctx.chunk = chunk
        ctx.set_materialize_grads(False)
        checkpoints = []
        with chunk.on_stream(args) as caller:
            chunk.inputs[-1].copy_(dynamics)
            for _ in range(_TICKS // _CHUNK):
                checkpoints.extend(state)
                chunk.advance(state)
                state = tuple(x.clone() for x in chunk.outputs)
            _record(state, caller)
        ctx.save_for_backward(dynamics, *checkpoints)
        return state

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *adjoints):
        chunk = ctx.chunk
        dynamics, *checkpoints = ctx.saved_tensors
        # The remaining checkpoints were allocated on the chunk's stream in forward.
        count = len(adjoints)
        with chunk.on_stream((*adjoints, dynamics, *checkpoints[:count])) as caller:
            chunk.inputs[-1].copy_(dynamics)
            context_grad = None
            for end in range(len(checkpoints), 0, -count):
                chunk.pullback(checkpoints[end - count:end], adjoints)
                adjoints = chunk.grads[:-1]
                context_piece = chunk.grads[-1]
                if context_piece is not None:
                    context_grad = context_piece.clone() if context_grad is None else context_grad.add_(context_piece)
            result = tuple(g.clone() if g is not None and need else None
                           for g, need in zip(adjoints, ctx.needs_input_grad[1:-1]))
            _record((*result, context_grad), caller)
        return (None, *result, context_grad)


def terminal_hold(predictor, state, dynamics):
    """Advance all particles for exactly 50 ticks; use only for terminal-state objectives."""
    tensors = (*state, dynamics)
    if (getattr(predictor, "_hold_graphs", None) is None or torch.is_inference_mode_enabled()
            or torch.is_autocast_enabled() or not dynamics.is_cuda
            or any(x.dtype != torch.float32 or torch.autograd.forward_ad.unpack_dual(x).tangent is not None
                   for x in tensors)):
        return _advance(predictor.model, state, dynamics.new_zeros(len(dynamics), 5), dynamics, _TICKS)
    cache = getattr(predictor, "_terminal_hold_chunks", None)
    if cache is None:
        cache = predictor._terminal_hold_chunks = {}
    key = (torch.cuda.current_stream(dynamics.device).cuda_stream, tuple(v.shape for v in state), dynamics.requires_grad)
    chunk = cache.get(key)
    if chunk is None:
        chunk = _Chunk(predictor.model, state, dynamics)
        if len(cache) >= 2:
            cache.pop(next(iter(cache)), None)
        cache[key] = chunk
    return _Hold.apply(chunk, *state, dynamics)
