"""Differentiable T1 landing prediction in the deployed model's world and joint frames.

Load a local bundle produced by experiments/export_t1_landing.py. TorchScript holds
one closed-loop tick. Perturbed command rollouts estimate STEP and HOLD
sensitivities, with native autodiff available on the nominal issue clocks. All poses below are sole-center (x, y, yaw), in m, m, rad.
"""

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

from cutamp.landing_accuracy import CHECKPOINT_SHA256, SOURCE_SHA256

SETTLING_TICKS = 25


@contextmanager
def prediction_precision():
    """Use the checkpoint evaluation's FP32 matmuls, restoring the planner's precision afterwards.

    Enclose a complete linearization, including autograd, when cuRobo has enabled TF32.
    """
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        # A profiling-mode HOLD can specialize the deployed tick before planning
        # and silently drop its input tangents on a subsequent batch shape. Use
        # the same ATen path for state preparation, FK and differentiation.
        with torch.jit.optimized_execution(False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _wrap(angle):
    # Preserve exact small-angle boundary inputs for the strict comparison.
    return torch.where(angle.abs() <= math.pi, angle, torch.atan2(angle.sin(), angle.cos()))


def landing_errors(achieved, reference):
    """Return per-foot radial XY/wrapped yaw errors and the joint tracking pass for each row."""
    d = achieved - reference
    xy = torch.linalg.vector_norm(d[..., :2], dim=-1)
    yaw = _wrap(d[..., 2]).abs()
    return xy, yaw, ((xy < 0.020) & (yaw < math.radians(1))).all(dim=-1)


def near_rest(max_joint_rms, max_root_speed):
    """Diagnostic over the last 25 HOLD ticks, independent of all acceptance masks."""
    return (max_joint_rms < 0.010) & (max_root_speed < 0.020)


def _sole_trajectory(model, physical):
    """Read world sole pose, height and tilt for every [particle, tick] state in one FK call."""
    batch, ticks = physical[0].shape[:2]
    outputs = model.feet(tuple(value.flatten(0, 1) for value in physical))
    return tuple(value.reshape(batch, ticks, *value.shape[1:]) for value in outputs)


class _HoldTickGraph:
    """Replay a single deployed tick, advancing each particle until its own deadline."""

    def __init__(self, model, state, dynamics, capacity):
        self.state = tuple(x.clone(memory_format=torch.contiguous_format) for x in state)
        self.dynamics = dynamics.clone(memory_format=torch.contiguous_format)
        self.index = torch.zeros(1, device=dynamics.device, dtype=torch.long)
        self.end = self.index.new_full((len(dynamics),), capacity)
        self.hold = dynamics.new_zeros(len(dynamics), 5)  # captured kernels retain this device address
        physical_and_action = (*state[:6], state[6][:, :state[4].shape[1]])
        self.trace = tuple(x.new_empty((capacity, *x.shape)) for x in physical_and_action)
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=dynamics.device)
        caller = torch.cuda.current_stream(dynamics.device)
        stream.wait_stream(caller)
        with torch.cuda.stream(stream):
            model(self.state, self.hold, self.dynamics)  # initialize kernels and the stream's BLAS workspace
            self.graph.capture_begin(capture_error_mode="thread_local")
            try:
                advanced = model(self.state, self.hold, self.dynamics)
                active = self.index < self.end
                for src, dst in zip(advanced, self.state):
                    dst.copy_(torch.where(active.view(-1, *([1] * (dst.ndim - 1))), src, dst))
                for dst, src in zip(self.trace, (*self.state[:6], self.state[6][:, :state[4].shape[1]])):
                    dst.index_copy_(0, self.index, src.unsqueeze(0))
                self.index.add_(1)
            finally:
                self.graph.capture_end()
        caller.wait_stream(stream)

    def reset(self, state, dynamics, counts):
        for dst, src in zip(self.state, state):
            dst.copy_(src)
        self.dynamics.copy_(dynamics)
        self.end.copy_(counts)
        self.index.zero_()

    def run(self, state, dynamics, counts, ticks):
        self.reset(state, dynamics, counts)
        for _ in range(ticks):
            self.graph.replay()
        # Each result outlives the next replay, including the single-particle case.
        terminal = tuple(x.clone() for x in self.state)
        trace = tuple(x[:ticks].transpose(0, 1).clone(memory_format=torch.contiguous_format) for x in self.trace)
        return terminal, trace[:6], trace[6]


def _hold_trace(predictor, state, dynamics, counts, ticks):
    """Return post-tick physical states and actions, preserving rows past their last valid tick."""
    cache = getattr(predictor, "_hold_graphs", None)
    if cache is not None and not torch.is_autocast_enabled() and all(
        x.dtype == torch.float32 and x.device == predictor.device and not x.requires_grad
        and torch.autograd.forward_ad.unpack_dual(x).tangent is None for x in (*state, dynamics)
    ):
        capacity = max(ticks, 62 + predictor.hold_ticks)
        shapes = tuple(x.shape for x in (*state, dynamics))
        key = (torch.cuda.current_stream(dynamics.device).cuda_stream, torch.is_inference_mode_enabled(), shapes, capacity)
        if key not in cache:
            if len(cache) >= 8:
                cache.pop(next(iter(cache)))
            cache[key] = _HoldTickGraph(predictor.model, state, dynamics, capacity)
        return cache[key].run(state, dynamics, counts, ticks)
    hold = dynamics.new_zeros(len(dynamics), 5)
    frames, actions = [], []
    for i in range(ticks):
        advanced = predictor.model(state, hold, dynamics)
        active = i < counts
        state = tuple(torch.where(active.view(-1, *([1] * (v.ndim - 1))), v, old) for v, old in zip(advanced, state))
        frames.append(state[:6])
        actions.append(state[6][:, :state[4].shape[1]])
    return state, tuple(torch.stack(series, dim=1) for series in zip(*frames)), torch.stack(actions, dim=1)


@dataclass(frozen=True)
class LandingProfile:
    """Phase tolerances in meters and radians for the model-predicted foot trace."""

    deadline_xy: float = 0.05
    deadline_yaw: float = 0.10
    deadline_height: float = 0.02
    support_xy: float = 0.01
    support_yaw: float = 0.05
    hold_xy: float = 0.02
    hold_yaw: float = 0.10

    def __post_init__(self):
        if any(v < 0 or not math.isfinite(v) for v in vars(self).values()):
            raise ValueError("Landing phase tolerances must be finite and nonnegative")


@dataclass
class PhaseChecks:
    deadline_xy: Tensor
    deadline_yaw: Tensor
    deadline_height: Tensor
    support_xy: Tensor
    support_yaw: Tensor
    hold_xy: Tensor  # peak drift for each sole from its own deadline pose
    hold_yaw: Tensor
    deadline_ok: Tensor
    support_ok: Tensor
    hold_ok: Tensor


def phase_checks(trace, heights, reference, side, clock, valid, profile):
    """The issued STEP produces frames 1..clock, then HOLD ends at the last valid frame."""
    rows = torch.arange(len(clock), device=clock.device)
    deadline = trace[rows, clock - 1]
    delta = deadline[:, side] - reference[:, side]
    arrival_xy = delta[:, :2].norm(dim=-1)
    arrival_yaw = _wrap(delta[:, 2]).abs()
    height = heights[rows, clock - 1, side]

    ticks = torch.arange(1, trace.shape[1] + 1, device=clock.device)
    during_step = ticks[None] <= clock[:, None]
    during_hold = valid & ~during_step
    planted = 1 - side
    slip = trace[:, :, planted] - reference[:, None, planted]
    support_xy = slip[..., :2].norm(dim=-1).masked_fill(~during_step, 0).amax(1)
    support_yaw = _wrap(slip[..., 2]).abs().masked_fill(~during_step, 0).amax(1)
    drift = trace - deadline[:, None]
    hold_xy = drift[..., :2].norm(dim=-1).masked_fill(~during_hold[:, :, None], 0).amax(1)
    hold_yaw = _wrap(drift[..., 2]).abs().masked_fill(~during_hold[:, :, None], 0).amax(1)
    return PhaseChecks(
        arrival_xy,
        arrival_yaw,
        height,
        support_xy,
        support_yaw,
        hold_xy,
        hold_yaw,
        (arrival_xy <= profile.deadline_xy)
        & (arrival_yaw <= profile.deadline_yaw)
        & (height < profile.deadline_height),
        (support_xy <= profile.support_xy) & (support_yaw <= profile.support_yaw),
        ((hold_xy <= profile.hold_xy) & (hold_yaw <= profile.hold_yaw)).all(-1),
    )


def _intended_pair(soles, side, command):
    support = soles[:, 1 - side]
    yaw = support[:, 2]
    x, y, turn = command.unbind(-1)
    swing = torch.stack(
        (
            support[:, 0] + yaw.cos() * x - yaw.sin() * y,
            support[:, 1] + yaw.sin() * x + yaw.cos() * y,
            _wrap(yaw + turn),
        ),
        dim=-1,
    )
    return torch.stack((swing, support) if side == 0 else (support, swing), dim=1)


@dataclass
class LandingState:
    """Root p,R,v,w, joints q,qd, 2-action queue, raw observation history, policy clock/anchors."""

    tensors: tuple[Tensor, ...]
    dynamics: Tensor  # normalized bfm_wm context
    history_source: str


@dataclass
class StepPrediction:
    side: str
    reference: Tensor
    achieved: Tensor
    clock: Tensor
    xy_error: Tensor
    yaw_error: Tensor
    tracking_ok: Tensor
    max_joint_rms: Tensor | None
    max_root_speed: Tensor | None
    near_rest: Tensor | None  # Requires a 25-tick post-landing HOLD window.
    terminal: LandingState
    # Batches can finish on different ticks. Padding holds the handoff state.
    trajectory: tuple[Tensor, ...]  # root p,R,v,w, joints q,qd; [B,T,...]
    sole_trace: Tensor
    sole_height: Tensor
    sole_tilt: Tensor
    actions: Tensor
    valid_ticks: Tensor
    phases: PhaseChecks


@dataclass
class LandingPrediction:
    steps: tuple[StepPrediction, ...]
    terminal: LandingState
    intended_final: Tensor
    final_xy_error: Tensor
    final_yaw_error: Tensor
    tracking_ok: Tensor  # precision only, full manipulation/motion feasibility is assessed by the caller


class _CommandJacobian(torch.autograd.Function):
    """Keep the deployed primal exactly. Its batch-independent command Jacobian is already known."""

    @staticmethod
    def forward(ctx, primal, tangent, commands):
        ctx.save_for_backward(tangent)
        ctx.command_shape = commands.shape
        return primal

    @staticmethod
    def backward(ctx, adjoint):
        (tangent,) = ctx.saved_tensors  # [particles, command coordinates, output coordinates]
        grad = torch.bmm(tangent, adjoint.reshape(len(adjoint), -1, 1)).reshape(ctx.command_shape)
        return None, None, grad


def _repeat_state(state, count):
    return LandingState(
        tuple(x.repeat_interleave(count, 0) for x in state.tensors),
        state.dynamics.repeat_interleave(count, 0),
        state.history_source,
    )


def _same_clocks(primal, sampled, count):
    return torch.stack(
        [
            (step.clock[:, None] == probe.clock.reshape(len(step.clock), count)).all()
            for step, probe in zip(primal.steps, sampled.steps, strict=True)
        ]
    ).all()


def _central_difference(samples, delta, angles):
    """Reduce [particle, sign × coordinate, ...] probes to command Jacobians."""
    plus, minus = samples.chunk(2, dim=1)
    diff = plus - minus
    if angles is not None:
        diff[..., angles] = _wrap(diff[..., angles])
    return diff.reshape(len(samples), len(delta), -1) / (2 * delta[:, None])


def _attach_prediction(primal, samples, order, profile, attach):
    """Attach command sensitivities to physical readouts, then recompute their nonlinear reductions."""
    connected = []
    for side, p, f in zip(order, primal.steps, samples.steps, strict=True):
        achieved = attach(p.achieved, f.achieved, 2)
        reference = attach(p.reference, f.reference, 2)
        trace = attach(p.sole_trace, f.sole_trace, 2)
        height = attach(p.sole_height, f.sole_height)
        # Physical state is p,R,v,w,q,qd. Extra terminal fields are action queue,
        # observation history, and an eight-component command latch ending in yaw.
        angle_columns = {4: slice(None), 7: slice(0, p.trajectory[4].shape[-1]), 8: 7}
        terminal = tuple(
            attach(x, y, angle_columns.get(i))
            for i, (x, y) in enumerate(zip(p.terminal.tensors, f.terminal.tensors, strict=True))
        )
        trajectory = tuple(
            attach(x, y, slice(None) if i == 4 else None)
            for i, (x, y) in enumerate(zip(p.trajectory, f.trajectory, strict=True))
        )
        xy, yaw, _ = landing_errors(achieved, reference)
        connected.append(
            replace(
                p,
                achieved=achieved,
                reference=reference,
                xy_error=xy,
                yaw_error=yaw,
                terminal=replace(p.terminal, tensors=terminal),
                trajectory=trajectory,
                sole_trace=trace,
                sole_height=height,
                sole_tilt=attach(p.sole_tilt, f.sole_tilt),
                actions=attach(p.actions, f.actions),
                phases=phase_checks(trace, height, reference, "LR".index(side), p.clock, p.valid_ticks, profile),
            )
        )
    intended = attach(primal.intended_final, samples.intended_final, 2)
    xy, yaw, _ = landing_errors(connected[-1].achieved, intended)
    return replace(
        primal,
        steps=tuple(connected),
        terminal=connected[-1].terminal,
        intended_final=intended,
        final_xy_error=xy,
        final_yaw_error=yaw,
    )


class LandingPredictor:
    def __init__(
        self,
        bundle: Path,
        device="cuda",
        hold_ticks=75,
        profile=None,
        *,
        expected_deployment=None,
        derivative_method="finite_difference",
        use_cuda_graph=False,
    ):
        if derivative_method not in ("finite_difference", "autodiff"):
            raise ValueError("derivative_method must be finite_difference or autodiff")
        self.derivative_method = derivative_method
        device = torch.device(device)
        if device.type != "cuda" or (device.index if device.index is not None else torch.cuda.current_device()) != 0:
            raise ValueError("The seed-3 bundle requires logical CUDA device 0 (its FK uses traced device constants)")
        if type(hold_ticks) is not int or hold_ticks < SETTLING_TICKS:
            raise ValueError("HOLD must cover the 25-tick diagnostic window")
        metadata = {"landing.json": ""}
        self.model = torch.jit.load(str(bundle), map_location=device, _extra_files=metadata).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.metadata = json.loads(metadata["landing.json"])
        if (
            self.metadata["format"] != 2
            or self.metadata["checkpoint_sha256"] != CHECKPOINT_SHA256
            or any(self.metadata["source_sha256"].get(name) != digest for name, digest in SOURCE_SHA256.items())
        ):
            raise ValueError(f"Incompatible T1 landing bundle: {bundle}")
        if expected_deployment is not None and self.metadata["deployment_sha256"] != expected_deployment:
            raise ValueError(f"T1 landing deployment fingerprint differs from the requested one: {bundle}")
        self.device = self.model.home_joints.device
        self.hold_ticks = hold_ticks
        self.profile = LandingProfile() if profile is None else profile
        self._hold_graphs = {} if use_cuda_graph else None

    @torch.no_grad()
    @prediction_precision()
    def standing_start(self, dynamics: Tensor) -> LandingState:
        """Explicit synthetic home fixture. Scene or post-retract starts must supply their own state."""
        if dynamics.ndim != 2 or dynamics.shape[1] != 38:
            raise ValueError("Expected [batch,38] normalized dynamics")
        dynamics = dynamics.to(device=self.device, dtype=torch.float32)
        batch = len(dynamics)
        q = self.model.home_joints.expand(batch, -1)
        p = q.new_zeros(batch, 3)
        p[:, 2] = 0.688
        R = torch.eye(3, device=q.device).expand(batch, -1, -1).contiguous()
        v = torch.zeros_like(p)
        qd = torch.zeros_like(q)
        previous = q.new_zeros(batch, 2 * q.shape[1])
        raw = torch.cat((qd, qd, v, -R[:, 2, :], qd), dim=-1)
        history = raw[:, None].repeat(1, self.metadata["history_ticks"], 1)
        tensors = (p, R, v, v, q, qd, previous, history, q.new_zeros(batch, 8))
        hold = q.new_zeros(batch, 5)
        for _ in range(50):
            tensors = self.model(tensors, hold, dynamics)
        return LandingState(tensors, dynamics, "synthetic_home_50_hold")

    def _step(self, initial, side, command, hold_ticks):
        physical = initial.tensors
        before = self.model.feet(physical[:6])[0]
        if not ((physical[-1][:, 0] > 0.5) & (physical[-1][:, 2:4] == 0).all(-1)).all():
            raise ValueError("A landing must be issued from an initialized HOLD")
        reference = _intended_pair(before, side, command)
        issue = torch.cat((torch.ones_like(command[:, :1]), torch.full_like(command[:, :1], side), command), dim=-1)
        state = self.model(physical, issue, initial.dynamics)
        clock = state[-1][:, 2].detach().to(torch.long) + 1
        if not ((clock >= 40) & (clock <= 62)).all():
            raise ValueError("The deployed clock did not accept the command")
        end = clock + hold_ticks
        times = torch.arange(1, int(end.max()) + 1, device=end.device)
        valid = times[None, :] <= end[:, None]
        issued = state
        state, following, actions = _hold_trace(self, issued, initial.dynamics, end - 1, len(times) - 1)
        trajectory = tuple(torch.cat((start[:, None], tail), dim=1) for start, tail in zip(issued, following))
        actions = torch.cat((issued[6][:, None, :issued[4].shape[1]], actions), dim=1)
        # These world-frame measurements are readouts. The policy's observation
        # feedback and command latch were already advanced inside each model tick.
        sole_trace, sole_height, sole_tilt = _sole_trajectory(self.model, trajectory)
        achieved = sole_trace[:, -1]
        xy, yaw, tracks = landing_errors(achieved, reference)
        joint_rms = root_speed = settled = None
        if hold_ticks >= SETTLING_TICKS:
            window = valid & (times[None, :] > end[:, None] - SETTLING_TICKS)
            with torch.no_grad():
                joint_rms = trajectory[5].square().mean(-1).sqrt().masked_fill(~window, 0).amax(1)
                root_speed = trajectory[2].norm(dim=-1).masked_fill(~window, 0).amax(1)
                settled = near_rest(joint_rms, root_speed)
        terminal = LandingState(state, initial.dynamics, initial.history_source)
        return StepPrediction(
            "LR"[side],
            reference,
            achieved,
            clock,
            xy,
            yaw,
            tracks,
            joint_rms,
            root_speed,
            settled,
            terminal,
            trajectory,
            sole_trace,
            sole_height,
            sole_tilt,
            actions,
            valid,
            phase_checks(sole_trace, sole_height, reference, side, clock, valid, self.profile),
        )

    @prediction_precision()
    def rollout(
        self, initial: LandingState, order: str, commands: Tensor, *, final_hold_ticks: int | None = None
    ) -> LandingPrediction:
        """Issue consecutive steps at their deadlines, then HOLD after the final step.

        A single-command replay can supply the recorded HOLD duration to reproduce its handoff.

        Use normal ATen autograd for the deployed tick. Torch 2.9's profiling
        executor can specialize the recurrent backward incorrectly when a fresh
        module's first invocation consumes an externally supplied state and a
        differentiable command. Its forward values then still match deployment.
        """
        if order == "none":  # exact-order name emitted by the search
            order = ""
        if order not in ("", "L", "R", "LR", "RL"):
            raise ValueError(f"Unsupported step order {order!r}")
        if final_hold_ticks is None:
            final_hold_ticks = self.hold_ticks
        if type(final_hold_ticks) is not int or final_hold_ticks < 0:
            raise ValueError("Expected a nonnegative integer final HOLD duration in ticks")
        if commands.shape != (len(initial.dynamics), len(order), 3) or not len(commands):
            raise ValueError("Expected nonempty [batch,steps,3] foot commands")
        if commands.dtype != torch.float32 or commands.device != self.device or not torch.isfinite(commands).all():
            raise ValueError("Expected finite float32 commands on the model device")
        if initial.dynamics.shape != (len(commands), 38) or len(initial.tensors) != 9:
            raise ValueError("Expected the complete T1 state, history, clock and dynamics")
        state = initial
        intended = achieved = self.model.feet(initial.tensors[:6])[0]
        accurate = torch.ones(len(commands), dtype=torch.bool, device=commands.device)
        steps = []
        for j, foot in enumerate(order):
            side = "LR".index(foot)
            intended = _intended_pair(intended, side, commands[:, j])
            hold_ticks = final_hold_ticks if j == len(order) - 1 else 0
            step = self._step(state, side, commands[:, j], hold_ticks)
            steps.append(step)
            state = step.terminal
            achieved = step.achieved
            accurate = accurate & step.tracking_ok
        xy, yaw, final_ok = landing_errors(achieved, intended)
        return LandingPrediction(tuple(steps), state, intended, xy, yaw, accurate & final_ok)

    def linearize(self, initial: LandingState, order: str, commands: Tensor) -> LandingPrediction:
        """Differentiate the fixed incoming state's command. All nominal values and gates are deployed values."""
        return self._linearize(initial, order, commands, self.derivative_method)

    def linearize_autodiff(self, initial: LandingState, order: str, commands: Tensor) -> LandingPrediction:
        """Reference method with three or six forward-mode directions through the frozen feedback recurrence."""
        return self._linearize(initial, order, commands, "autodiff")

    @prediction_precision()
    def _linearize(self, initial, order, commands, method):
        # The nominal grad-enabled rollout uses the exact kernels of the original
        # differentiated call and gives a correct reverse-mode fallback at clock branches.
        primal = self.rollout(initial, order, commands)
        if not commands.requires_grad or not torch.is_grad_enabled() or not primal.steps:
            return primal
        if any(x.requires_grad for x in (*initial.tensors, initial.dynamics)):
            return primal  # A command-only Jacobian would discard incoming-state derivatives.
        batch, steps, coords = commands.shape
        directions = steps * coords
        basis = torch.eye(directions, device=commands.device, dtype=commands.dtype)
        if method == "finite_difference":
            # Each particle gets its own +/- perturbation of every command. An LR/RL
            # probe carries its own achieved first stance and controller history into step two.
            delta = commands.new_tensor([0.001, 0.001, math.radians(0.1)]).repeat(steps)
            offsets = torch.cat((basis * delta, -basis * delta)).reshape(2 * directions, steps, coords)
            with torch.no_grad():
                probes = (commands[:, None] + offsets).reshape(-1, steps, coords)
                samples = self.rollout(_repeat_state(initial, 2 * directions), order, probes)
                clocks_match = _same_clocks(primal, samples, 2 * directions)
            if clocks_match:
                finite = []

                def attach_fd(value, samples, angles=None):
                    if not value.requires_grad:
                        return value
                    with torch.no_grad():
                        probes = samples.reshape(batch, 2 * directions, *value.shape[1:])
                        jac = _central_difference(probes, delta, angles)
                        finite.append(torch.isfinite(jac).all())
                    return _CommandJacobian.apply(value.detach(), jac, commands)

                connected = _attach_prediction(primal, samples, order, self.profile, attach_fd)
                if not finite or torch.stack(finite).all():
                    return connected
                del connected
            del samples

        # A probe crossed an issue clock, its slope was nonfinite, or exact AD was requested.
        # Packed AD follows the nominal branch locally; FP32 batching can very rarely
        # change a nominal clock too, in which case retain the original reverse graph.
        with torch.autograd.forward_ad.dual_level():
            with torch.no_grad():
                tangents = basis.repeat(batch, 1).reshape(batch * directions, steps, coords)
                packed = torch.autograd.forward_ad.make_dual(
                    commands.detach().repeat_interleave(directions, 0), tangents
                )
                forward = self.rollout(_repeat_state(initial, directions), order, packed)
                if not _same_clocks(primal, forward, directions):
                    return primal

            def attach_ad(value, dual, angles=None):
                tangent = torch.autograd.forward_ad.unpack_dual(dual).tangent
                if tangent is None:
                    return value  # Constant, or a specialized TorchScript field retaining native reverse mode.
                return _CommandJacobian.apply(value.detach(), tangent.reshape(batch, directions, -1), commands)

            return _attach_prediction(primal, forward, order, self.profile, attach_ad)
