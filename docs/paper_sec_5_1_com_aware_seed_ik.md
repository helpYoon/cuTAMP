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

Let $c(q)=(c_x(q),c_y(q))$ be the robot's mass-weighted CoM projected onto the horizontal plane and expressed in the `mobile_base_link` frame. The support rectangle is centered at the frame origin and has half-widths $h_x$ and $h_y$. We first define how far the CoM lies beyond each support boundary,

$$
d_x(q)=\max\!\left(|c_x(q)|-h_x,0\right),
\qquad
d_y(q)=\max\!\left(|c_y(q)|-h_y,0\right).
$$

Both values are zero while the CoM remains inside the rectangle. To create a soft buffer of width $m$ near its edges, define

$$
d_{\mathrm{margin}}(q)=\max\!\left(
|c_x(q)|-h_x+m,
|c_y(q)|-h_y+m,
0
\right).
$$

The implemented CoM penalty is then

$$
P_{\mathrm{com}}(q)=d_x(q)^2+d_y(q)^2
+w_{\mathrm{in}}d_{\mathrm{margin}}(q)^2
+w_c\left[
\left(\frac{c_x(q)}{h_x}\right)^2+
\left(\frac{c_y(q)}{h_y}\right)^2
\right].
\tag{5}
$$

The first two terms penalize motion outside the support rectangle. The margin term becomes positive when the CoM enters the $m$-wide strip near any edge and remains active outside the rectangle. The final term gently biases otherwise feasible postures toward the center. The implemented values are $h_x=0.1115\,\mathrm{m}$, $h_y=0.156\,\mathrm{m}$, $m=0.02\,\mathrm{m}$, $w_{\mathrm{in}}=1$, $w_c=10^{-3}$, and an overall support weight $w=1$.

The other two terms in the seed objective come from cuRobo. Let $\mathcal{L}$ be the set of tracked frames: the active hand, inactive hand, and base. For each frame $\ell\in\mathcal{L}$, let $p_\ell(q)$ and $R_\ell(q)$ be its current position and orientation, and let $p_\ell^d$ and $R_\ell^d$ be the corresponding targets. If $\phi(\cdot)$ converts a relative rotation to its axis--angle vector, then the current unit-weight pose score is

$$
E_{\mathrm{pose}}(q)=\sum_{\ell\in\mathcal{L}}
\left[
\frac{1}{2}\left\lVert p_\ell(q)-p_\ell^d\right\rVert_2^2
+\left\lVert\phi\!\left(R_\ell(q)R_\ell^{d\top}\right)\right\rVert_2^2
\right].
$$

Thus, $E_{\mathrm{pose}}$ is small only when every tracked frame is close to its target in both position and orientation.

For joint $i$, let $q_i^-$ and $q_i^+$ denote the lower and upper bounds. cuRobo moves each original bound inward by $0.1\%$ of that joint's range before LM begins. Define the amount by which the joint violates these margin-adjusted bounds as

$$
v_i(q)=\max(q_i^- - q_i,0)+\max(q_i-q_i^+,0).
$$

With the current unit joint-limit weight, cuRobo forms

$$
r_{\mathrm{lim}}(q)=\begin{bmatrix}v_1(q)&\cdots&v_n(q)\end{bmatrix}^{\top},
\qquad
E_{\mathrm{lim}}(q)=\sum_{i=1}^{n}v_i(q).
$$

Each $v_i$ is zero inside the permitted interval and equals the distance beyond the violated bound outside it. The scalar $E_{\mathrm{lim}}$ is used when comparing the current and candidate objectives. The local LM system separately treats $r_{\mathrm{lim}}$ as a residual, producing the term $J_{\mathrm{lim}}^\top r_{\mathrm{lim}}$ in the update below.

The complete scalar objective used by the gain-ratio test is

$$
E(q)=E_{\mathrm{pose}}(q)+E_{\mathrm{lim}}(q)+wP_{\mathrm{com}}(q).
\tag{6}
$$

Let $K$ denote the number of LM candidates retained for later stages. Figure 5.1 summarizes where the modified objective enters the end-to-end solve and, separately, how one LM candidate is accepted or rejected.

![Two-panel CoM-aware IK pipeline. Panel (a) shows batched three-frame goals, the 64-candidate LM seed stage, retention of K candidates, the all-success early-return branch, optional CoM-blind particle-to-L-BFGS refinement, and downstream CoM checks. Panel (b) shows current-state evaluation, the damped trial solve, candidate evaluation, and gain-ratio acceptance; the CoM contribution enters the objective and gradient but adds no Jacobian rows.](ik_pipeline.svg)

*Figure 5.1: Two-panel CoM-aware IK pipeline. Teal marks the CoM additions in the seed stage; amber marks CoM-blind refinement. The post-IK CoM checks are outside cuRobo's success predicate. $B$ is the configured batch capacity and $K$ is the number of LM candidates retained for later stages.*

## Mapping the objective to the implemented solver

Let $J_{\mathrm{lim}}=\partial r_{\mathrm{lim}}/\partial q$ be the analytical joint-limit Jacobian, and let $J_{\mathrm{pose}}$ be the tool-pose Jacobian. The implementation forms the right-hand side gradient $g$ and residual Jacobian $J$ as

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
