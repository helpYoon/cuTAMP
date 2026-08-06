# 5.1 Center-of-Mass-Aware Seed Inverse Kinematics

<!-- Equation numbers are local to this section and must be reconciled when merged into the paper.
     Markers of the form [xref: ...] must likewise be replaced by the paper's final cross-references. -->

## Why pose-only IK is insufficient

One target comprises the active hand's sampled task pose and current-pose targets for the inactive hand and mobile base. The base's three virtual planar joints are locked, leaving the active configuration $q\in\mathbb{R}^{18}$; the locked values are reinserted into the returned 21-DoF configuration. These hand and base pose constraints do not uniquely choose the remaining whole-body posture. Torso, knee, and ankle motion can change the center of mass (CoM) without materially changing the task pose, so pose-only inverse kinematics has no reason to prefer a stable posture within that redundant set [xref: motivation section].

Let $B$ denote the configured batch capacity, and hence the number of targets processed together after input padding. This capacity follows `num_particles` and currently defaults to 256; 256 is a memory-conscious configuration value, not an algorithmic requirement.

## Levenberg–Marquardt from first principles

Forward kinematics defines a smooth map

$$
x=f(q), \qquad q\in\mathbb{R}^n,\quad x\in\mathbb{R}^m,
$$

where $q$ contains the joint variables and $x$ represents the tracked task-space poses. Inverse kinematics seeks a configuration for which $f(q)=x_d$, where $x_d$ is the desired stacked pose target. Because $f$ is nonlinear and generally has no closed-form inverse, IK is instead formulated as nonlinear least squares. Define the pose residual and its weighted cost as

$$
e(q)=x_d\ominus f(q),
\qquad
E_0(q)=\frac{1}{2}e(q)^\top W_e e(q).
\tag{1}
$$

Here $\ominus$ denotes the appropriate pose difference, including both translation and orientation, and $W_e\succeq0$ weights the components of that error. Defining the weighted residual $r(q)=W_e^{1/2}e(q)$ gives $E_0(q)=\tfrac12\lVert r(q)\rVert_2^2$. LM is the standard nonlinear least-squares solver applied to this cost.

For an increment $\delta q$, let $J(q)=\partial r/\partial q$ be the residual Jacobian at the current configuration. Its first-order expansion is

$$
r(q+\delta q)\approx r(q)+J(q)\delta q.
\tag{2}
$$

Gauss--Newton minimizes the squared norm of this local linearized residual. Setting the resulting quadratic model's gradient to zero gives

$$
J^\top J\,\delta q=-J^\top r.
\tag{3}
$$

In whole-body IK, $J^\top J$ can be singular or poorly conditioned along redundant directions. Levenberg--Marquardt (LM) introduces a positive damping parameter $\lambda$ and the identity matrix $I$, and instead solves

$$
(J^\top J+\lambda I)\delta q=-J^\top r.
\tag{4}
$$

Positive damping makes the linear solve well posed. Small $\lambda$ approaches the Gauss--Newton step, whereas increasing $\lambda$ interpolates toward the shorter gradient step $-J^\top r/\lambda$.

$E_0$ is the pose-only objective underlying this LM step. The implementation follows the same structure but supplies its native pose and joint-limit score, gradient, and Jacobian blocks separately.

## Adding the support-polygon objective

Let $c(q)\in\mathbb{R}^2$ be the mass-weighted robot CoM projected into the `mobile_base_link` frame. Let $h\in\mathbb{R}^2$ contain the rectangular support half-extents, let $[x]_+=\max(x,0)$ and $|x|$ act componentwise, let $m$ be an inside margin, let $w_{\mathrm{in}}$ and $w_c$ be the inside-barrier and center weights, and let $\oslash$ denote componentwise division. The implemented penalty is

$$
P_{\mathrm{com}}(q)=
\left\lVert[|c(q)|-h]_+\right\rVert_2^2
+w_{\mathrm{in}}
\left\lVert[|c(q)|-h+m]_+\right\rVert_\infty^2
+w_c\left\lVert c(q)\oslash h\right\rVert_2^2.
\tag{5}
$$

The first term is zero inside the support rectangle and grows quadratically outside it. The second activates within $m$ of the nearest edge and remains active outside, creating a soft interior buffer. The normalized center term breaks ties among otherwise feasible redundant postures by biasing the CoM toward the rectangle center. The implemented geometry and weights are $h=(0.1115,0.156)\,\mathrm{m}$, $m=0.02\,\mathrm{m}$, $w_{\mathrm{in}}=1$, $w_c=10^{-3}$, and an overall support weight $w=1$.

Using $E_{\mathrm{pose}}$ and $E_{\mathrm{lim}}$ for cuRobo's native pose and joint-limit terms, respectively, the modified seed objective is

$$
E(q)=E_{\mathrm{pose}}(q)+E_{\mathrm{lim}}(q)+wP_{\mathrm{com}}(q).
\tag{6}
$$

Let $K$ denote the number of LM candidates retained for later stages. Figure 5.1 summarizes where the modified objective enters the end-to-end solve and, separately, how one LM candidate is accepted or rejected.

![Two-panel CoM-aware IK pipeline. Panel (a) shows batched three-frame goals, the 64-candidate LM seed stage, retention of K candidates, the all-success early-return branch, optional CoM-blind particle-to-L-BFGS refinement, and downstream CoM checks. Panel (b) shows current-state evaluation, the damped trial solve, candidate evaluation, and gain-ratio acceptance; the CoM contribution enters the objective and gradient but adds no Jacobian rows.](ik_pipeline.svg)

*Figure 5.1: Two-panel CoM-aware IK pipeline. Teal marks the CoM additions in the seed stage; amber marks CoM-blind refinement. The post-IK CoM checks are outside cuRobo's success predicate. $B$ is the configured batch capacity and $K$ is the number of LM candidates retained for later stages.*

## Mapping the objective to the implemented solver

Let $r_{\mathrm{lim}}$ and $J_{\mathrm{lim}}$ be the analytical joint-limit residual and Jacobian, respectively, and let $J_{\mathrm{pose}}$ be the tool-pose Jacobian. The implementation forms the right-hand side gradient $g$ and residual Jacobian $J$ as

$$
g=\nabla_q\!\left(E_{\mathrm{pose}}+wP_{\mathrm{com}}\right)
+J_{\mathrm{lim}}^\top r_{\mathrm{lim}},
\qquad
J=\begin{bmatrix}J_{\mathrm{pose}}\\J_{\mathrm{lim}}\end{bmatrix},
\tag{7}
$$

One fused forward-kinematics evaluation emits the tool poses, $J_{\mathrm{pose}}$, and robot CoM. One combined autograd backward then adds the pose and CoM gradients safely, while the joint-limit gradient and Jacobian are assembled analytically. Consequently, the CoM term changes $E$ and $g$ but contributes no rows to $J$.

With these separately assembled quantities, each candidate step solves

$$
(J^\top J+\lambda I)\delta q=-g.
\tag{8}
$$

One batched Warp launch forms and Cholesky-solves an $18\times18$ system for every flattened problem. After this damped trial solve, the candidate $q+\delta q$ is evaluated before the gain-ratio decision. The implemented ratio is

$$
\rho=
\frac{E(q)-E(q+\delta q)}
{\tfrac12\delta q^\top(\lambda\delta q-g)}.
\tag{9}
$$

The step is accepted when $\rho\ge 10^{-5}$. Accepted rows use the candidate and halve $\lambda$; rejected rows retain the current iterate and double $\lambda$. Damping starts at $\lambda=1$ and is clamped to $[10^{-5},10^{10}]$.

RHS-only coupling is useful because pose-null directions have weak $J^\top J$ curvature. Damping therefore allows the CoM gradient to choose among redundant postures without adding a new residual or Jacobian template. Candidate re-evaluation nevertheless tests the full modified score, limiting steps whose omitted CoM curvature makes the local model inaccurate.

## Seed selection, conditional refinement, and output checks

For each solve, $B$ targets times 64 candidates gives $64B$ independent LM problems, each executing 30 LM iterations. Each target receives one supplied current-state seed and 63 bounded Halton seeds. The supplied state is the home configuration for initial particle IK and the current particle configuration $q$ for periodic coupled re-IK.

After LM, successful candidates rank before failed candidates. Within that split, the ranking score is position error plus rotation error plus $0.01$ times distance to the current state. Seed success means that the maximum tracked-frame position error is below $5\,\mathrm{mm}$, the maximum orientation error is below $0.05\,\mathrm{rad}$, and every active joint is inside its limits. The CoM penalty is not part of this Boolean seed-success test. The full solver retains $K=32$ candidates, while periodic coupled re-IK retains $K=4$.

Standard metrics then evaluate pose convergence and feasibility, including collision. If all $B$ targets pass, the best candidate is returned without refinement. Otherwise, the retained candidates initialize particle optimization and then L-BFGS, after which the metrics are recomputed. Collision is absent from seed LM, whereas CoM is absent from this refinement and from the final success predicate. The returned active solution is finally expanded back to 21 DoF by reinserting the locked joints.

The CoM term is therefore not itself a pass/fail condition, but it can still alter convergence. Conservative tuning avoided an observed regression in the validation sweep; this does not imply that IK success rates are invariant to the added objective.

## Why the soft objective and hard gate are both needed

The soft term changes preference, not admissibility. It guides seed LM toward better postures within a redundant pose-solution set, but no finite soft weight proves static stability. Moreover, refinement and the final success predicate are CoM-blind. The cuTAMP post-IK mask and retry therefore remain necessary, and the later `ComPolygon` constraint remains the hard stability enforcement [xref: layered-enforcement section]. These layers keep CoM bias, retry behavior, collision feasibility, and hard stability enforcement distinct.

## Parameter selection

Empirical sweeps selected the support weight $w=1$ within a robust tested range of $0.1$--$3.0$, together with center weight $w_c=10^{-3}$. Center weights at or above $3\times10^{-3}$ degraded seed convergence and increased reliance on CoM-blind refinement. These values are empirical tuning results rather than a theorem or stability guarantee.

---

<!-- Merge these entries into the paper bibliography. -->
**References.**
[1] K. Levenberg, "A method for the solution of certain non-linear problems in least
squares," *Quarterly of Applied Mathematics*, 2(2):164--168, 1944.
[2] D. W. Marquardt, "An algorithm for least-squares estimation of nonlinear
parameters," *J. SIAM*, 11(2):431--441, 1963.
[3] K. Madsen, H. B. Nielsen, and O. Tingleff, *Methods for Non-Linear Least Squares
Problems*, 2nd ed., IMM, Technical University of Denmark, 2004.
