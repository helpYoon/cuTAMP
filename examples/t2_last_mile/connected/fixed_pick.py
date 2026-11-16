"""Evaluate native fixed-grasp IK particles when symbolic Holding has no Adam variables."""
def install():
 from pathlib import Path
 import inspect,cutamp.optimize_plan as module
 cls=module.ParticleOptimizer;src=inspect.getsource(cls.__call__)
 old='''        if not pairs:
            raise RuntimeError(f"No parameters to optimize! For plan skeleton: {plan_skeleton}")'''
 new='''        if not pairs:
            timer.stop("setup_optimizer")
            with torch.no_grad():
                cost_dict = cost_fn(rollout_fn(particles), particles)
                mask = self.get_satisfying_mask(cost_dict, verbose=True)
                costs = self.cost_reducer(cost_dict, consider_types=ADAM_CONSIDER_TYPES)
                soft = self.cost_reducer.soft_costs(cost_dict)
            n = int(mask.sum().item())
            opt_metrics.update(num_satisfying_final=n, num_steps=0,
                fixed_parameter_feasibility=True,
                best_cost=float(costs[mask].min()) if n else None,
                best_soft_cost=float(soft[mask].min()) if n else None)
            return n > 0, opt_metrics, False'''
 assert old in src
 namespace={};exec(src.replace(old,new).replace('    def __call__','def __call__',1) if False else __import__('textwrap').dedent(src.replace(old,new)),module.__dict__,namespace);cls.__call__=namespace['__call__']
