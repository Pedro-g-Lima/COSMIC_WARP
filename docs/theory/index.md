# Theory

The models and methods behind every step, each tied to the code that
implements it. Start with the conventions: every other page follows them.

- [Frames, state and units](conventions.md) — environments, bodies and
  entities; the inertial and body frames; the momentum state; units and
  precision.
- [Rigid-body dynamics](dynamics.md) — translational and rotational equations
  of motion, quaternion kinematics, variable mass and burnout.
- [Gravity & J2](gravity.md) — point-mass gravity, the J2 perturbation,
  gravity-gradient torque.
- [Environment models](environment.md) — dipole magnetic field and torque,
  atmospheric drag.
- [Actuator models](actuators.md) — reaction wheels and thrusters.
- [Numerical integration](integrators.md) — Euler, RK4 and symplectic
  kick–drift–kick; what limits their order.
- [Initial-condition sampling](sampling.md) — Cartesian boxes, orbital
  elements, Kepler's equation, attitude distributions.
- [GPU execution and the RL step](execution.md) — the fused step kernel, CUDA
  graphs, termination, resets, and the RL step contract.
- [Differentiable dynamics](differentiable.md) — what the reverse-mode
  gradients differentiate, and where they are not smooth.

Each model is checked against analytic and reference solutions on the
[Validation](../validation/index.md) pages; the sources are in the
[bibliography](../references.md).
