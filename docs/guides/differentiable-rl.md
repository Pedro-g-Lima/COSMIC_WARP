# Differentiable RL

The [trainer adapters](rl.md#trainer-adapters) treat the simulator as a
black box. The differentiable path backpropagates through the dynamics
instead, so a policy can be trained with first-order gradients — for
example with [SHAC](https://arxiv.org/abs/2204.07137) (Short-Horizon
Actor-Critic, Xu et al. 2022).

The package provides the differentiable environment in three torch-coupled
modules (they need the `torch` extra, and `cosmic_warp.rl` does not import
them):

| Module | Provides |
| --- | --- |
| [`diff_bridge`](../reference/api/cosmic_warp/rl/diff_bridge.md) | `DiffSimStepper`: one simulator step as a `torch.autograd.Function` on flat float64 tensors, state `[E, X_DIM]` and commands `[E, U_DIM]`, with the step's Warp adjoint as its backward |
| [`diff_tasks`](../reference/api/cosmic_warp/rl/diff_tasks.md) | the `DiffTask` protocol: action mapping, observation and reward as torch functions of the state |
| [`diff_env`](../reference/api/cosmic_warp/rl/diff_env.md) | `DiffVecEnv`: H-step differentiable windows, the rollout unit of a first-order trainer |

Tasks and trainers are application code.
[Example 17](../examples/17_rl_shac_training.md) brings both in
`examples/shac.py` — `AttitudeDiffTask`, a `DiffTask` mirroring
`AttitudeTask`, and a minimal `SHAC` trainer; copy and adapt them. Its contract
is in [the RL step contract](../theory/execution.md#differentiable-path).

## Train

```python
import tempfile
from pathlib import Path

from cosmic_warp.core.pipeline import SpacecraftSimulator
from cosmic_warp.rl.diff_env import DiffVecEnv

from shac import SHAC, AttitudeDiffTask          # examples/shac.py

sim = SpacecraftSimulator(
    num_envs=32, integrator="symplectic", dt=0.1,
    differentiable=True, check_bounds=False,     # both required
    reset_config={"seed": 0},
)
sim.vehicle.set_config(mass_dry=12.0, inertia=[[0.4, 0, 0], [0, 0.4, 0], [0, 0, 0.3]])
sim.rw_manager.set_config(max_torque=0.02, max_momentum=1.0, wheel_inertia=0.005)
sim.reset_manager.set_bounds(att_max_angle=0.5236, ang_mom_bounds=((-0.2,) * 3, (0.2,) * 3),
                             mass_bounds=(12.0, 12.0))

env = DiffVecEnv(sim, AttitudeDiffTask(w_att=0.0, w_rate=1.0, w_ctrl=1.0e-3),
                 horizon=16, max_episode_steps=96)
shac = SHAC(env, {"seed": 0, "critic_epochs": 2})   # overrides of default_shac_cfg()
history = shac.train(num_updates=2)                 # example 17: 50 updates of 4096 envs

obs = env.reset()                                   # [N, obs_dim] float64
action = shac.policy(obs, deterministic=True)

checkpoint = Path(tempfile.mkdtemp()) / "shac.pt"
shac.save(checkpoint)                               # networks, optimizers, obs statistics
shac.load(checkpoint)                               # same dimensions and network sizes
```

`train()` returns one stats dict per update (mean reward, losses, gradient
norm, policy std). The checkpoint includes the update count, so a resumed
`train()` continues the linear learning-rate decay instead of restarting it.
The cfg `seed` seeds torch (network initialization, exploration noise,
critic minibatches); the simulator's resets are seeded by
`reset_config={"seed": ...}`. `default_shac_cfg()` in `examples/shac.py`
lists the tunables.

## Windows

![One DiffVecEnv window: H steps of one environment from the carried, detached state; a policy shared by every step maps observations to actions; rewards go into the trainer's loss, and backward runs through every step back to the start of the window](../assets/learning-light.svg#only-light)
![One DiffVecEnv window: H steps of one environment from the carried, detached state; a policy shared by every step maps observations to actions; rewards go into the trainer's loss, and backward runs through every step back to the start of the window](../assets/learning-dark.svg#only-dark)

/// caption
One window, forward and backward. `backward()` gives the gradient of the
loss with respect to the policy parameters θ through every step of the
window; how a trainer uses it is the trainer's business. Shown for one
environment: all N run in the same launches, with one θ.
///

`DiffVecEnv` advances in windows of `horizon` steps:

- `env.rollout(policy_fn)` runs one window from the carried state and
  returns its observations (`horizon + 1`) and rewards (`horizon`), float64
  and on the autograd graph. `policy_fn` maps observations to actions in
  `[-1, 1]` and may be anything torch can differentiate.
- `env.end_window()` detaches and carries the final state, writes it back
  to the simulator, and resets every environment when the episode time
  limit is reached (it then returns `True`).
- `env.state` is the carried state `[E, X_DIM]` the next window starts from.

A window's backward uses the vehicle, thruster and wheel configurations of
its forward pass, copied once per window. `loss.backward()` may therefore
come before or after `end_window()`, even when `end_window()` resets the
environments and redraws randomized parameters, or after a `set_config`.

Calling `rollout()` again before `end_window()` re-runs the same window,
which is useful for finite-difference checks; `end_window()` without a
pending rollout, or after `reset()`, raises `RuntimeError`.

Episodes are truncation-only, because termination checks are off:
`max_episode_steps` must be a multiple of `horizon`, and all environments
reset together on that boundary, outside autograd.

## How the SHAC example uses the gradient

Each step is one autograd node, so a window builds the chain
`obs_t → a_t = π_θ(obs_t) → x_{t+1} → obs_{t+1} → …`. The actor loss

$$ L_\theta = -\tfrac{1}{NH} \sum_n \Big[ \sum_{t<H} \gamma^t r_t + \gamma^H V'(o_H) \Big] $$

backpropagates through the rewards, the dynamics, the policy at every step —
including how an action changes later observations and hence later actions —
and the target critic's terminal value. The detached carry between windows
truncates backpropagation at the window boundary; the critic, regressed on
TD(λ) targets from the same windows and tracked by a Polyak-averaged target
network, accounts for the rest.

The actor is a tanh-squashed Gaussian with a learnable per-dimension
standard deviation, reparameterized so the exploration noise is
differentiable. Networks are float32; the dynamics stay float64, with one
cast at the policy boundary. `tests/test_rl_diff_env.py` checks the bridge
and `DiffVecEnv` gradients with `torch.autograd.gradcheck` and finite
differences; `tests/test_example_shac.py` checks the example's task and
trainer. In its slow test, SHAC cuts the mean tumble rate by at least 40%
in 30 updates; the PPO adapter tests are given 100–150 updates for the same
threshold ([validation](../validation/learning.md#shac-example)).

## Requirements and limits

- **`differentiable=True` and termination checks off** (`check_bounds=False`,
  `check_planet_impact=False`); `DiffSimStepper` raises otherwise. A
  terminated entity stops being integrated, which would cut its gradient
  path.
- **No CUDA graph on the taped path.** Each step launches its kernels and
  allocates its own tape and buffers, so a differentiable step costs more
  than a captured `VecEnv` step. Under `torch.no_grad()`, or when no input
  requires a gradient, the stepper uses an untaped path with reused buffers.
  The [Benchmarks](../benchmarks.md) page lists measured window rates.
- **Memory grows with the horizon**: every step node keeps its buffers until
  backward. A taped `DiffSimStepper.step()` also keeps its own copy of the
  vehicle, thruster and wheel configurations; steps inside
  `with stepper.frozen_params():`, as in a `DiffVecEnv` window, share one
  copy taken on entry, so do not change the configuration inside the block.
- **Torch on its default stream.** Warp launches on its own blocking CUDA
  stream, which torch's default stream synchronizes with implicitly. If you
  run torch on a side stream (`torch.cuda.Stream()`, non-blocking),
  synchronize around `step()` yourself, or run Warp on that stream with
  `wp.stream_from_torch`.
- **First-order only.** The step's backward is a Warp adjoint, so double
  backward (Hessian-vector products, gradient penalties through the
  dynamics) raises `RuntimeError`.
- **Clamps stop gradients.** Wheel torque and momentum limits have zero
  gradient in saturation. The example's policy squashes actions with `tanh`
  and `AttitudeDiffTask` only scales them; keep task action paths
  clamp-free.
- **Smooth task math.** `AttitudeDiffTask` replaces the attitude angle's
  `acos`, whose derivative is infinite at the goal, with a bounded `atan2`
  form.
- **Constants cached in `bind` must stay constant.** `bind` runs once;
  `AttitudeDiffTask` caches the inverse inertia there, so it rejects inertia
  randomization, which redraws the inertia at every reset.

## SHAC or PPO

First-order training suits smooth dynamics and rewards, such as attitude
control and orbital maneuvers, and needs no external RL framework. Prefer
the [PPO adapters](rl.md#trainer-adapters) when the reward or dynamics are
discontinuous, when episodes must terminate early, or when you want an
established trainer and its tooling.
