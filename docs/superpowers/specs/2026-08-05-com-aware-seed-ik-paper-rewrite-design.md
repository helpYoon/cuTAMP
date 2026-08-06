# CoM-aware seed-IK paper section rewrite — design

**Date:** 2026-08-05

**Status:** approved

**Scope:** `docs/paper_sec_5_1_com_aware_seed_ik.md`, `docs/ik_pipeline.svg`, and one LM state-consistency correction with a focused regression test

## Goal

Rewrite Section 5.1 as a self-contained paper section that lets a reader without prior Levenberg–Marquardt (LM) knowledge understand:

1. why pose-only humanoid IK leaves stability-relevant posture undetermined;
2. how Gauss–Newton and LM obtain a joint-space update;
3. where the implemented center-of-mass (CoM) penalty enters that update;
4. how the complete cuRobo seed-search, selection, refinement, and feasibility pipeline behaves; and
5. why a soft IK bias is retained alongside a downstream hard CoM check.

The prose will distinguish canonical LM from cuRobo's concrete pose and joint-limit terms instead of claiming that the implementation uses an exact residual objective it does not use.

## Verified implementation to describe

### Inputs and dimensions

- Each target contains three tracked frames: the active hand receives the task pose; the inactive hand and `mobile_base_link` receive their current poses.
- The three planar-base joints are locked, so the IK search variable has 18 active joints; the locked joints are reinserted in the returned 21-DoF configuration.
- Let `B` denote the requested target batch. Inputs are padded to the configured capacity when needed. The capacity follows `num_particles`, whose current default is 256; the paper must not call it a fixed batch of 512.

### Seed stage

- Three tracked frames select the implemented 64-candidate, 30-iteration seed-IK schedule.
- A supplied current state occupies one seed row and the remaining rows are bounded Halton samples. The initial particle solve supplies the home state; coupled re-IK supplies the particle's current configuration.
- The `B x 64` problems are flattened and evaluated in parallel.
- After LM, each target retains `K` solutions. `K=32` for particle initialization and final full solves; `K=4` for periodic coupled re-IK.
- Seed ranking prioritizes pose/joint-limit success, then position plus rotation error and a `0.01`-weighted distance to the supplied current state.

### Conditional refinement and result

- The retained configurations are checked by the standard metrics rollout, which includes pose convergence and constraint feasibility, including collision.
- If all `B` targets pass, the solver returns directly from the seed candidates.
- Otherwise, the retained candidates initialize the particle optimizer followed by L-BFGS, after which metrics are recomputed.
- The refinement and final success predicate contain no CoM term. CoM can therefore change which basin seed IK supplies, but it neither directly vetoes a returned solution nor guarantees feasibility.
- cuTAMP subsequently applies its CoM mask/retry layer, and the later `ComPolygon` constraint remains the hard stability check.

## Mathematical presentation

The section will first derive canonical LM from a generic least-squares IK problem:

\[
\min_q \tfrac12\lVert r(q)\rVert^2,
\qquad
r(q+\delta q)\approx r(q)+J\delta q,
\]

leading to Gauss–Newton and then the damped LM system

\[
(J^\top J+\lambda I)\delta q=-J^\top r.
\]

It will then map that derivation to the implementation using names for the actual native terms, `E_pose` and `E_lim`, and define the support penalty from the base-frame CoM projection `c(q)`:

\[
P_{\mathrm{com}}(q)=
\lVert[|c|-h]_+\rVert_2^2
+w_{\mathrm{in}}\lVert[|c|-h+m]_+\rVert_\infty^2
+w_c\lVert c/h\rVert_2^2.
\]

The implemented values are `h=(0.1115, 0.156) m`, `m=0.02 m`, `w_in=1`, `w_c=10^-3`, and overall support weight `w=1`.

The fork computes pose and CoM contributions in one autograd backward through the same fused FK result. The assembled right-hand side is

\[
g=\nabla_q(E_{\mathrm{pose}}+wP_{\mathrm{com}})+J_{\mathrm{lim}}^\top r_{\mathrm{lim}},
\]

while the matrix remains

\[
J^\top J+\lambda I,
\qquad J=[J_{\mathrm{pose}};J_{\mathrm{lim}}].
\]

Thus the CoM term is gradient-only, or RHS-only: it contributes to the evaluated objective and `g`, but adds no Jacobian rows or second-order curvature. The gain-ratio test evaluates the candidate with the CoM term included and adapts damping per problem.

The explanation will motivate this choice: pose-null directions have weak `J^T J` curvature, so the CoM gradient can resolve redundant posture, while damping and candidate acceptance limit steps that harm the full objective. It will not claim that this soft priority preserves success rates by construction; only the reported sweep supports the chosen weights empirically.

## Figure design

Replace the current nested diagram with one paper-width SVG containing two labeled panels.

### Panel (a): complete solve

Show this exact branch structure:

`B three-frame targets -> 64 LM candidates x 30 iterations -> retain K -> pose/collision metrics -> all B pass?`

- yes: select and return the best configuration;
- no: particle optimizer -> L-BFGS -> recompute metrics -> select and return;
- after either branch: cuTAMP CoM mask/retry and downstream hard CoM constraint.

Annotate `K=32` for full solves and `K=4` for periodic re-IK. Keep batch size symbolic and note the current default `B=256` in prose, not as a supposedly fixed algorithmic constant.

### Panel (b): one LM iteration

Show the actual order:

1. evaluate the current `q` with fused FK, producing poses, `J_pose`, and CoM;
2. assemble objective, `g`, and `J`;
3. solve the damped system to obtain candidate `q+`;
4. evaluate `q+` with the same FK/objective path;
5. accept or reject by gain ratio, update `lambda`, and continue.

Teal marks only the fork's CoM path. Amber marks stages that are deliberately CoM-blind. Labels, arrows, and shape differences must remain understandable without relying on color. The caption will explain the conditional refinement branch and the separation between soft bias and hard verification.

## Manuscript structure

1. Problem and contribution.
2. Pose-only IK and humanoid redundancy.
3. Gauss–Newton and LM from first principles.
4. Implemented CoM penalty and RHS-only coupling.
5. Two-panel pipeline figure and stage-by-stage implementation mapping.
6. Seed selection, conditional refinement, and success semantics.
7. Rationale for retaining the downstream hard gate.
8. Parameter choices and empirical tuning boundary.

The rewrite will remove code-review language, oversized captions, stale constants, and unsupported claims such as “the penalty cannot change success rates.” Existing bibliography placeholders and cross-reference placeholders remain local because paper-wide merge work is outside scope.

## LM state-consistency correction

The current rejection path retains the old joint position, Jacobian, and gradient but stores the rejected candidate's `error_norm`. The next trust ratio therefore compares against an objective that does not correspond to the retained iterate.

Make the minimum correction in `seed_iteration_state_manager.py`: select `error_norm` from the candidate for accepted rows and from the current state for rejected rows. Extend the existing state-manager test file with one mixed accepted/rejected regression test that verifies the selected objective values alongside the already-selected state fields. No new abstraction or test file is needed.

## Verification

- Run the focused state-manager test module.
- Run any directly affected CoM-aware seed-IK tests that are available without changing test infrastructure.
- Parse the SVG as XML and inspect a rendered image for clipping, arrow direction, and readable labels.
- Cross-check every value and branch in the prose and caption against the solver/configuration files.
- Review the Markdown for undefined notation, equation/figure ordering, stale `512` claims, unsupported guarantees, and unresolved placeholders other than the explicitly retained paper-merge cross-references.

## Out of scope

- Reworking other paper sections or the paper-wide bibliography.
- Changing the CoM penalty, weights, seed schedule, refinement configuration, or success thresholds.
- Refactoring cuRobo's optimizer or introducing a CoM Jacobian/Hessian block.
