import torch
from torch.optim import Adam

from cutamp import optimize_plan as op
from cutamp.config import TAMPConfiguration
from cutamp.cuda_graphs import CapturedGraph
from cutamp.optimize_plan import _finite_step
from cutamp.tests.conftest import clone_particles, make_plan_info, needs_cuda, run_optimizer


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


def test_the_countdown_reads_the_published_cost_at_start_weights():
    from cutamp.cost_reduction import CostReducer
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    published = {"at_start": torch.tensor([0.5, float("nan"), 0.25])}

    def cost_fn(rollout, particles):
        soft = {"type": "cost", "values": {"preference": x}}
        if published["at_start"] is not None:
            soft["cost_at_start_weights"] = published["at_start"]
        return {"Row": {"type": "constraint", "values": {"row": 100 * x}, "weights": {"row": torch.full((3,), 8.0)}},
                "soft": soft}

    body = op._StepBody({"x": x}, lambda particles: None, cost_fn, CostReducer({}),
                        lambda cost, verbose: torch.zeros(3, dtype=torch.bool), op.ADAM_CONSIDER_TYPES, params=[x],
                        grad_masks={}, keep_vis=False, stepped=True)
    assert body()[0][2] == 0.25                     # the published total, its row that is not a number skipped
    published["at_start"], x.grad = None, None
    assert body()[0][2] == 801.0                    # without it, the weighted total, as before


def test_a_step_that_is_not_judged_leaves_the_record_alone():
    """A recorded step takes the record of valid particles at every replay, through the flag the loop sets."""
    from cutamp.cost_reduction import CostReducer
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    def cost_fn(rollout, particles):
        return {"soft": {"type": "cost", "values": {"preference": x}, "rank": torch.zeros(3), "effort": x.detach()}}

    archive = op._FeasibleArchive({"x": x}, torch.zeros(3), torch.zeros(3))
    body = op._StepBody({"x": x}, lambda particles: None, cost_fn, CostReducer({}),
                        lambda cost, verbose: torch.tensor([True, False, True]), op.ADAM_CONSIDER_TYPES, params=[x],
                        grad_masks={}, keep_vis=False, archive=archive, stepped=True, judge=torch.tensor(False))
    body()
    assert archive.center.isinf().all() and archive.effort.isinf().all()   # not judged: nothing is kept
    body.judge.fill_(True)
    x.grad = None
    body()
    inf = float("inf")
    assert archive.center.tolist() == [0.0, inf, 0.0] and archive.effort.tolist() == [1.0, inf, 3.0]


class FakeSteps:
    """What the loop asks of a stepped plan, around a plan without steps: its body runs on ``rollout_fn`` after
    ``put_back`` and before ``hand_on``, and a recording of it guards what ``recorded`` names."""
    landing_confs, gap_of = frozenset(), {}

    def __init__(self, info, world):
        particles = info["particles"]
        self.command_names = {0: "step_commands_0"}
        particles["step_commands_0"] = torch.zeros(len(particles["left_q0"]), 1, 3, device="cuda")
        self.rollout_fn, self.world = info["rollout_fn"], world
        self.refreshes, self.put_backs, self.stats = 0, 0, {}
        self.capped = torch.zeros((), dtype=torch.long, device="cuda")

    def put_back(self, particles):
        self.put_backs += 1
        self.capped.add_(1)

    def hand_on(self, particles):
        pass

    def recorded(self):
        return [self.capped], [self.world.site_kinematics(site) for site in ("rollout", "retract", "com", "retract_motion")]

    def refresh(self, particles, slopes=True):
        self.refreshes += 1

    def step(self, optimizer, params, particles):
        return _finite_step(optimizer, params)


@needs_cuda
def test_the_loop_refreshes_judges_and_keeps_every_posture(real_world, real_plan, monkeypatch):
    world, (skeleton, base) = real_world, real_plan  # a plan without steps, driven through the hook by FakeSteps
    config = TAMPConfiguration(num_particles=64, num_opt_steps=12, reik_interval=5, early_stop_patience=0,
                               cuda_graphs=True, enable_visualizer=False)
    info = make_plan_info(world, skeleton, clone_particles(base), config)
    steps = info["steps"] = FakeSteps(info, world)
    judged, replay = [], CapturedGraph.replay

    def consulted(*a):
        raise AssertionError("the posture re-solve was consulted")

    def spy(self):
        if self.name == "G_step":
            judged.append(bool(self.body.judge))
        return replay(self)

    monkeypatch.setattr(CapturedGraph, "replay", spy)
    monkeypatch.setattr(op, "_build_ik_deps", consulted)
    metrics, _ = run_optimizer(info, config)
    assert steps.refreshes == 3            # after steps 4 and 9, then the last one after the restore
    assert steps.put_backs == 4            # in Python: three warm-ups and the recording, which runs nothing
    assert int(steps.capped) == 12         # on the device: once before each Adam step's body, the warm-ups undone
    assert [k for k, on in enumerate(judged) if on] == [0, 5, 10]   # the record is taken on these replays only
    assert metrics["capture_step"] == 0 and "best_com_margin" not in metrics
    assert metrics["steps"] is steps.stats
    particles = info["particles"]
    assert particles["step_commands_0"].requires_grad
    pick_posture = next(c for c in info["rollout_fn"].conf_params if c not in ("q0", "left_q0", "right_q0"))
    assert particles[pick_posture].requires_grad  # not handed to the re-solve


@needs_cuda
def test_a_cost_that_publishes_a_rank_keeps_its_record_and_its_countdown_by_it(real_world, real_plan, monkeypatch):
    """The record and the stop rule read the rank and effort a stepped plan's cost publishes, never its centring pair
    (2026-11-23 design 4.5 and 4.6)."""
    world, (skeleton, base) = real_world, real_plan
    config = TAMPConfiguration(num_particles=64, num_opt_steps=12, reik_interval=5, early_stop_patience=0,
                               cuda_graphs=False, enable_visualizer=False)  # eager: the test reads each call
    info = make_plan_info(world, skeleton, clone_particles(base), config)
    info["steps"] = FakeSteps(info, world)
    inner, published = info["cost_fn"], []

    class Ranked:
        """The plan's cost, which publishes a centring pair, with a rank and effort of its own beside it."""
        world = inner.world

        def __call__(self, rollout, particles):
            cost = inner(rollout, particles)
            soft = cost["soft"]
            assert "max_center_distance_m" in soft
            soft["rank"] = 4096.0 + torch.arange(len(soft["max_center_distance_m"]), device="cuda")
            soft["effort"] = torch.full_like(soft["rank"], 0.5)
            published.append((soft["rank"], soft["effort"]))
            return cost

    info["cost_fn"] = Ranked()
    made, observed, by_rank = [], [], []
    init, observe, should_stop = op._FeasibleArchive.__init__, op._FeasibleArchive.observe, op._BestSnapshot.should_stop
    monkeypatch.setattr(op._FeasibleArchive, "__init__",
                        lambda self, p, c, e: (made.append((c, e)), init(self, p, c, e))[1])
    monkeypatch.setattr(op._FeasibleArchive, "observe",
                        lambda self, p, f, c, e: (observed.append((c, e)), observe(self, p, f, c, e))[1])
    monkeypatch.setattr(op._BestSnapshot, "should_stop",
                        lambda self, *a, **k: (by_rank.append(k.get("by_rank")), should_stop(self, *a, **k))[1])
    run_optimizer(info, config)
    assert len(made) == 1 and len(observed) == 12 and by_rank == [True] * 3   # a record per step, masked unless judged
    for center, effort in made + observed:
        assert any(center is rank and effort is worst for rank, worst in published)
