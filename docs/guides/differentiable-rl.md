# Differentiable RL

The [trainer adapters](rl.md#trainer-adapters) treat the simulator as a black
box. The differentiable path backpropagates through the dynamics instead, so
a policy can be trained with first-order gradients, for example with
[SHAC](https://arxiv.org/abs/2204.07137) (Short-Horizon Actor-Critic, Xu et
al. 2022).

Three torch-coupled modules provide it (they need the `torch` extra;
`cosmic_warp.rl` does not import them):

| Module | Provides |
| --- | --- |
| [`diff_bridge`](../reference/api/cosmic_warp/rl/diff_bridge.md) | `DiffSimStepper`: one step as a `torch.autograd.Function` on flat float64 tensors, state `[E, X_DIM]` and commands `[E, U_DIM]`, with the Warp adjoint as backward |
| [`diff_tasks`](../reference/api/cosmic_warp/rl/diff_tasks.md) | the `DiffTask` protocol: action mapping, observation and reward as torch functions of the state |
| [`diff_env`](../reference/api/cosmic_warp/rl/diff_env.md) | `DiffVecEnv`: H-step differentiable windows, the rollout unit of a first-order trainer |

Tasks and trainers are application code: `examples/shac.py`
([example 17](../examples/17_rl_shac_training.md)) has `AttitudeDiffTask`, a
`DiffTask` mirroring `AttitudeTask`, and a minimal `SHAC` trainer to copy
and adapt. Contract: [differentiable path](../theory/execution.md#differentiable-path).

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

`train()` returns one stats dict per update. The checkpoint holds the
update count, so a resumed `train()` continues the learning-rate decay. The
cfg `seed` seeds torch (initialization, exploration, critic minibatches);
`reset_config={"seed": ...}` seeds the resets. `default_shac_cfg()` in
`examples/shac.py` lists the tunables.

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

| Call | Does |
| --- | --- |
| `env.rollout(policy_fn)` | runs one window from the carried state; returns its observations (`horizon + 1`) and rewards (`horizon`), float64, on the autograd graph. `policy_fn` maps observations to actions in `[-1, 1]`. |
| `env.end_window()` | detaches and carries the final state, writes it back to the simulator, and resets every environment at the episode time limit (then returns `True`) |
| `env.state` | the carried state `[E, X_DIM]` the next window starts from |

A window's backward uses the configurations of its forward pass, copied once
per window, so `loss.backward()` may come before or after `end_window()`,
even when that resets and redraws randomized parameters. Calling `rollout()`
again before `end_window()` re-runs the window (useful for finite
differences); `end_window()` without a pending rollout, or after `reset()`,
raises `RuntimeError`.

Episodes are truncation-only, because termination checks are off:
`max_episode_steps` is a multiple of `horizon`, and all environments reset
together on that boundary, outside autograd.

## How the SHAC example uses the gradient

Each step is one autograd node, so a window chains
`obs_t → a_t = π_θ(obs_t) → x_{t+1} → obs_{t+1} → …`. The actor loss

$$ L_\theta = -\tfrac{1}{NH} \sum_n \Big[ \sum_{t<H} \gamma^t r_t + \gamma^H V'(o_H) \Big] $$

backpropagates through the rewards, the dynamics and the policy at every
step, and the target critic's terminal value. The detached carry truncates
backpropagation at the window boundary; the critic, regressed on TD(λ)
targets and tracked by a Polyak-averaged target network, covers the rest.
The actor is a tanh-squashed Gaussian with learnable per-dimension standard
deviation, reparameterized so the exploration noise is differentiable.
Networks are float32; the dynamics stay float64, cast once at the policy
boundary. Evidence: [validation](../validation/learning.md#torch-bridge).

## Requirements and limits

The [rollout limits](differentiable.md#requirements-and-limits) apply:
`differentiable=True`, termination checks off (`DiffSimStepper` raises
otherwise), memory growing with the horizon, no CUDA graph on taped steps.
In addition:

- **Cost.** Each taped step allocates its own tape and buffers; under
  `torch.no_grad()`, or when no input requires a gradient, the stepper takes
  an untaped path with reused buffers. Window rates: [Benchmarks](../benchmarks.md).
- **Parameters.** A taped `DiffSimStepper.step()` keeps its own copy of the
  configurations; steps inside `with stepper.frozen_params():`, as in a
  window, share one copy taken on entry, so do not change the configuration
  inside the block.
- **Torch on its default stream.** Warp launches on its own blocking stream,
  which torch's default stream synchronizes with. On a non-blocking torch
  side stream, synchronize around `step()` yourself, or run Warp on it with
  `wp.stream_from_torch`.
- **First-order only**: double backward raises `RuntimeError`.
- **Clamp-free, smooth task math.** Clamps have zero gradient in saturation:
  the example squashes actions with `tanh` and the task only scales them.
  `AttitudeDiffTask` replaces the attitude angle's `acos`, whose derivative
  is infinite at the goal, with a bounded `atan2` form.
- **Constants cached in `bind` stay constant.** `AttitudeDiffTask` caches
  the inverse inertia, so it rejects inertia randomization.

## SHAC or PPO

First-order training suits smooth dynamics and rewards, such as attitude
control and orbital manoeuvres, and needs no RL framework. Prefer the
[PPO adapters](rl.md#trainer-adapters) for discontinuous rewards or
dynamics, early termination, or an established trainer's tooling. In the
slow tests SHAC cuts the tumble rate by 40 % in 30 updates, where the PPO
adapters get 100–150 ([validation](../validation/learning.md#shac-example)).
