import torch
from torch.optim import Adam

from cutamp import optimize_plan as op
from cutamp.cuda_graphs import CapturedGraph
from cutamp.optimize_plan import _finite_step
from cutamp.tests.conftest import needs_cuda, plan_info, run_optimizer, zero_step_plan


def test_a_row_without_a_finite_gradient_keeps_its_values():
    a, b = torch.zeros(3, 2, requires_grad=True), torch.zeros(3, 4, requires_grad=True)
    unused = torch.zeros(3, 1, requires_grad=True)  # no cost reads it: its gradient is None
    optimizer = Adam([a, b, unused], lr=0.1)
    a.grad, b.grad = torch.ones(3, 2), torch.ones(3, 4)
    b.grad[1, 2] = float("nan")
    held = _finite_step(optimizer, [a, b, unused])
    assert held.tolist() == [False, True, False]
    assert (a[1] == 0).all() and (b[1] == 0).all() and (unused == 0).all()
    assert (a[[0, 2]] != 0).all() and (b[[0, 2]] != 0).all()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()


class _Still:
    """A plan whose put-back and hand-on move nothing."""

    def put_back(self, particles):
        pass

    def hand_on(self, particles):
        pass


def test_the_countdown_reads_the_published_cost_at_start_weights():
    from cutamp.cost_reduction import CostReducer
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    published = {"at_start": torch.tensor([0.5, float("nan"), 0.25])}

    def cost_fn(rollout, particles):
        soft = {"type": "cost", "values": {"preference": x}, "score": torch.zeros(3),
                "rank": torch.zeros(3), "effort": torch.zeros(3)}
        if published["at_start"] is not None:
            soft["cost_at_start_weights"] = published["at_start"]
        return {"Row": {"type": "constraint", "values": {"row": 100 * x}, "weights": {"row": torch.full((3,), 8.0)}},
                "soft": soft}

    body = op._StepBody({"x": x}, lambda particles: None, cost_fn, CostReducer({}),
                        lambda cost, verbose: torch.zeros(3, dtype=torch.bool), op.ADAM_CONSIDER_TYPES, params=[x],
                        grad_masks={}, keep_vis=False, steps=_Still(),
                        archive=op._FeasibleArchive({"x": x}, *torch.zeros(3, 3)), judge=torch.tensor(True))
    assert body()[0][2] == 0.25                     # the published total, its row that is not a number skipped
    published["at_start"], x.grad = None, None
    assert body()[0][2] == 801.0                    # without it, the weighted total, as before


def test_a_step_that_is_not_judged_leaves_the_record_alone():
    """A recorded step takes the record of valid particles at every replay, through the flag the loop sets."""
    from cutamp.cost_reduction import CostReducer
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    def cost_fn(rollout, particles):
        return {"soft": {"type": "cost", "values": {"preference": x}, "score": torch.zeros(3),
                         "rank": torch.zeros(3), "effort": x.detach()}}

    archive = op._FeasibleArchive({"x": x}, *torch.zeros(3, 3))
    body = op._StepBody({"x": x}, lambda particles: None, cost_fn, CostReducer({}),
                        lambda cost, verbose: torch.tensor([True, False, True]), op.ADAM_CONSIDER_TYPES, params=[x],
                        grad_masks={}, keep_vis=False, steps=_Still(), archive=archive, judge=torch.tensor(False))
    body()
    assert archive.score.isinf().all() and archive.effort.isinf().all()   # not judged: nothing is kept
    body.judge.fill_(True)
    x.grad = None
    body()
    inf = float("inf")
    assert archive.score.tolist() == [0.0, inf, 0.0] and archive.effort.tolist() == [1.0, inf, 3.0]


class FakeSteps:
    """What the loop asks of a stepped plan, around a plan without steps (:func:`zero_step_plan`): its body runs on the
    plan's ``rollout_fn`` after ``put_back`` and before ``hand_on``, and a recording of it guards what ``recorded``
    names; it adds one command, which no cost reads, and counts the put-backs and refreshes."""
    landing_confs, gap_of = frozenset(), {}

    def __init__(self, plan):
        particles = plan.particles
        self.command_names = {0: "step_commands_0"}
        particles["step_commands_0"] = torch.zeros(len(particles["left_q0"]), 1, 3, device="cuda")
        self.plan, self.rollout_fn = plan, plan.rollout_fn
        self.refreshes, self.put_backs, self.stats = 0, 0, {}
        self.capped = torch.zeros((), dtype=torch.long, device="cuda")

    def put_back(self, particles):
        self.put_backs += 1
        self.capped.add_(1)

    def hand_on(self, particles):
        pass

    def recorded(self):
        tensors, kinematics = self.plan.recorded()
        return [self.capped, *tensors], kinematics

    def refresh(self, particles, slopes=True):
        self.refreshes += 1

    def step(self, optimizer, params, particles):
        return _finite_step(optimizer, params)


@needs_cuda
def test_the_loop_refreshes_judges_and_keeps_every_posture(real_world, monkeypatch):
    config = dict(num_opt_steps=12, reik_interval=5, early_stop_patience=0, cuda_graphs=True, enable_visualizer=False)
    plan = zero_step_plan(real_world, **config)
    info = plan_info(plan)
    steps = info["steps"] = FakeSteps(plan)
    judged, replay = [], CapturedGraph.replay

    def spy(self):
        if self.name == "G_step":
            judged.append(bool(self.body.judge))
        return replay(self)

    monkeypatch.setattr(CapturedGraph, "replay", spy)
    metrics, _ = run_optimizer(info, plan.config)
    assert steps.refreshes == 3            # after steps 4 and 9, then the last one after the restore
    assert steps.put_backs == 4            # in Python: three warm-ups and the recording, which runs nothing
    assert int(steps.capped) == 12         # on the device: once before each Adam step's body, the warm-ups undone
    assert [k for k, on in enumerate(judged) if on] == [0, 5, 10]   # the record is taken on these replays only
    assert metrics["capture_step"] == 0
    assert metrics["steps"] is steps.stats
    particles = info["particles"]
    assert particles["step_commands_0"].requires_grad
    assert all(particles[name].requires_grad for name in plan.gap_of)   # every posture stays an Adam variable


@needs_cuda
def test_the_record_and_the_countdown_read_the_rank_the_cost_publishes(real_world, monkeypatch):
    """The record and the stop rule read the score, rank and effort the plan's cost publishes (2026-11-23
    design 4.5 and 4.6)."""
    config = dict(num_opt_steps=12, reik_interval=5, early_stop_patience=0, cuda_graphs=False,
                  enable_visualizer=False)  # eager: the test reads each call
    plan = zero_step_plan(real_world, **config)
    info = plan_info(plan)
    info["steps"] = FakeSteps(plan)
    published = []

    def cost(rollout, particles):
        out = plan(rollout, particles)
        published.append(tuple(out["soft"][k] for k in ("score", "rank", "effort")))
        return out

    cost.world = plan.world  # what the optimizer reads off its cost
    info["cost_fn"] = cost
    made, observed, stops = [], [], []
    init, observe, should_stop = op._FeasibleArchive.__init__, op._FeasibleArchive.observe, op._BestSnapshot.should_stop
    monkeypatch.setattr(op._FeasibleArchive, "__init__",
                        lambda self, p, *keys: (made.append(keys), init(self, p, *keys))[1])
    monkeypatch.setattr(op._FeasibleArchive, "observe",
                        lambda self, p, f, *keys: (observed.append(keys), observe(self, p, f, *keys))[1])
    monkeypatch.setattr(op._BestSnapshot, "should_stop",
                        lambda self, *a: (stops.append(a[0]), should_stop(self, *a))[1])
    run_optimizer(info, plan.config)
    assert len(made) == 1 and len(observed) == 12 and stops == [0, 5, 10]   # a record per step, masked unless judged
    for keys in made + observed:
        assert any(all(a is b for a, b in zip(keys, row, strict=True)) for row in published)
