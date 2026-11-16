# Release validation

Validated on Linux x86_64 with CPython 3.12.3, MuJoCo 3.8.1, NumPy 2.2.6, SciPy 1.15.3 and ProxSuite 0.8.2. Pytest 9.0.2 was installed into an isolated test-tools directory; the qualified physics environment was not changed.

- Asset-free contract tests: **3 passed**. They cover robot-only JSON initialization preserving scene objects, packet/contact derivatives, and refusal of an incompatible body mapping.
- Frozen versus cleaned reference packets: maximum absolute difference **0**, across ten times spanning empty steps, payload stepping and manipulation.
- Frozen successful controller versus cleaned controller: five states spanning double support, left/right swings, loaded swing and deep floor reach. Maximum torque difference **5.39e-11 N m**. Hard constraints and numerical objectives are unchanged. No extra physics rollout was needed for this parity check.
- The execution package separately qualifies the full 113.112 s continuous task using the cleaned interfaces. See its evidence/admission documentation; recorded binary traces and videos are not committed here.

The cleaned controller adds explicit T2 interface guards and makes rejected-QP dumps opt-in. `StepReference` materializes NPZ arrays once. The component CLI supports XML or version-matched MJB input. The unified generator accepts portable named-state JSON, carries passive robot joints, explicitly preserves gripper holding mode, and exposes support-point placement without changing physical sole geometry.

Portability constraints remain explicit: the controller assumes the T2's 31 body joints, named ankle/gripper frames, first free trunk root, fixed 500 Hz control, and supplied sole geometry. External robot/gripper assets and a compatible physical model are required; use the execution package's verified model builder and asset inventory. The CPU runtime is separate from the planner's Python 3.10/CUDA environment. No hard-real-time or hardware safety qualification is claimed.

Planner integration audit: the authoritative optimizer/anatomical/GN/stance/export sources are supplied by the planner overlay, not copied historical snapshots here. No host absolute paths were found in its production Python/YAML sources. Its relative launcher path adjustments are repository-relative. The custom optimizer remains explicitly labeled; it must not be presented as stock cuRobo success.
