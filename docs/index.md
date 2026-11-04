# COSMIC WARP

<div class="cw-hero" markdown>

![COSMIC WARP](assets/banner.png)

**GPU-parallel spacecraft dynamics on [NVIDIA Warp](https://github.com/NVIDIA/warp).**
Thousands to millions of spacecraft step in parallel, in float64, with each
`simulate()` loop replayed as one CUDA graph.

```bash
pip install cosmic-warp
```

</div>

COSMIC WARP is the simulator, not the application: guidance, control and
training algorithms are yours.

## What you get

- **6-DOF dynamics** with variable mass, in float64; Euler, RK4 (default) or
  symplectic kick–drift–kick.
- **[Physics models](guides/physics.md)**: point-mass gravity, J2,
  atmospheric drag, gravity-gradient and magnetic torques, each switchable.
- **[Actuators](guides/actuators.md)**: up to 8 thrusters and 4 reaction
  wheels per spacecraft.
- **Batches on the device**: one or several spacecraft per environment,
  sampled resets, domain randomization and termination, all inside the
  captured step.
- **[Scenarios in YAML](guides/configuration.md)**, strictly checked.
- **Learning**: a vectorized [RL environment](guides/rl.md) with adapters for
  rsl_rl, skrl and rl_games, and [gradients](guides/differentiable.md)
  through the dynamics.
- **[Validation](validation/index.md)**: every claim names the test that
  checks it, against analytic results where they exist.

## Where to start

<div class="grid cards" markdown>

-   :material-rocket-launch-outline:{ .lg .middle } **Getting started**

    ---

    Install, propagate a batch, fire a thruster, load a YAML scenario.

    [:material-arrow-right: Start here](getting-started.md)

-   :material-book-open-variant:{ .lg .middle } **Guides**

    ---

    One task per page: vehicles, actuators, resets, randomization, physics,
    integrators, scaling, RL and gradients.

    [:material-arrow-right: Browse the guides](guides/index.md)

-   :material-file-code-outline:{ .lg .middle } **Examples**

    ---

    Seventeen runnable scripts, from a basic orbit to Monte Carlo campaigns,
    gradient-based optimization and RL training.

    [:material-arrow-right: Browse examples](examples/index.md)

-   :material-check-decagram-outline:{ .lg .middle } **Validation**

    ---

    Every claim with the test that checks it.

    [:material-arrow-right: See the evidence](validation/index.md)

-   :material-function-variant:{ .lg .middle } **Theory**

    ---

    The equations behind the kernels, and the frames, state and units every
    number follows.

    [:material-arrow-right: Read the theory](theory/index.md)

-   :material-api:{ .lg .middle } **API reference**

    ---

    Generated from the docstrings of every module in `cosmic_warp`.

    [:material-arrow-right: Open the reference](reference/api/cosmic_warp/index.md)

</div>
