r"""SHAC trainer and differentiable attitude task

SHAC (Short-Horizon Actor-Critic, Xu et al. 2022) and the differentiable
attitude task used by example 17. It is a module imported by
`17_rl_shac_training.py`, not a script to run.

First-order policy learning: the actor loss

$$
L_\theta = -\frac{1}{NH} \sum_n \Big[ \sum_{t<H} \gamma^t r_t + \gamma^H V'(o_H) \Big]
$$

is differentiated **analytically through the simulator**: the H-step window
from `cosmic_warp.rl.diff_env.DiffVecEnv` keeps the dynamics, the reward and
the policy in the loop on one autograd graph (Warp adjoints inside each step
node). The critic V is trained by regression on TD(λ) targets from the same
windows (no simulator gradient), and a Polyak-averaged target critic V'
provides the bootstrap values.

The actor is a tanh-squashed Gaussian with a learnable per-dimension std,
reparameterized so that the gradient flows through the sample. The networks
run in float32; the dynamics stay float64 and are cast at the network
inputs and outputs.

`AttitudeDiffTask` mirrors `AttitudeTask` from `attitude_task.py` in torch:
the same actions, observations and reward, except that the attitude angle
uses a form whose gradient stays finite at the goal attitude. Inertia
randomization is not supported.

This is a minimal reference implementation, not part of the package: copy
and adapt it. The package provides the differentiable environment
(`DiffVecEnv`, the `DiffTask` protocol, the torch bridge).

```python
from shac import SHAC, AttitudeDiffTask   # examples/ is the script's directory

env = DiffVecEnv(sim, AttitudeDiffTask(), horizon=32, max_episode_steps=192)
shac = SHAC(env)
stats = shac.train(num_updates=100)
```
"""
import warnings
from copy import deepcopy
from typing import Callable, Optional

import torch
import torch.nn as nn

from cosmic_warp.rl.diff_env import DiffVecEnv

#: Under the safe-angle sqrt(|q_v|^2 + eps) the reward matches the true
#: attitude angle to O(sqrt(eps)) while keeping the gradient finite (and zero)
#: exactly at the goal quaternion.
_ANGLE_EPS = 1e-16


class AttitudeDiffTask:
    """Differentiable reaction-wheel attitude control (mirrors
    `AttitudeTask` in `attitude_task.py`).

    Same action/observation/reward semantics as the warp-kernel task — actions
    command wheels 0-2 scaled by the assembly's `max_torque`, observation is
    per body `[quat(4), angvel(3), rw_ang_mom(4)]` with the quaternion
    canonicalized to the `q_w >= 0` hemisphere, reward is
    `-(w_att*angle^2 + w_rate*|omega|^2 + w_ctrl*|a|^2)` — except the
    attitude angle uses the numerically safe `2*atan2(sqrt(|q_v|^2 + eps),
    |q_w|)`, whose gradient stays finite (and zero) at the goal.

    `bind` snapshots the inverse inertia and torque limits: configure the
    simulator first. Inertia randomization is rejected.
    """

    ACT_PER_BODY = 3
    OBS_PER_BODY = 11

    def __init__(self, w_att: float = 1.0, w_rate: float = 0.1, w_ctrl: float = 0.01):
        self.w_att = float(w_att)
        self.w_rate = float(w_rate)
        self.w_ctrl = float(w_ctrl)

    def bind(self, sim) -> None:
        self._num_envs = sim.num_envs
        self._num_bodies = sim.num_bodies
        self.act_dim = sim.num_bodies * self.ACT_PER_BODY
        self.obs_dim = sim.num_bodies * self.OBS_PER_BODY

        # The inverse inertia is cached here, but inertia randomization
        # redraws it at every reset: omega would come from a stale tensor.
        if any("inertia_scale" in sim.randomizer.get_bounds(b) for b in range(sim.num_bodies)):
            raise ValueError(
                "AttitudeDiffTask caches the inertia at bind and does not "
                "support inertia randomization; remove inertia_scale from "
                "the randomizer bounds."
            )
        device = str(sim.device)
        vehicles = sim.vehicle.configs.numpy()
        rw = sim.rw_manager.configs.numpy()
        self._inertia_inv = torch.tensor(
            vehicles["inertia_inv"], dtype=torch.float64, device=device)   # [E, 3, 3]
        self._max_torque = torch.tensor(
            rw["max_torque"], dtype=torch.float64, device=device)          # [E]
        if (rw["spin_axis_3"] != 0.0).any():
            warnings.warn(
                "AttitudeDiffTask commands only wheels 0-2; the configured 4th "
                "reaction wheel is observed but never commanded.",
                stacklevel=2,
            )

    def commands(self, action: torch.Tensor) -> torch.Tensor:
        E = self._num_envs * self._num_bodies
        torque = action.reshape(E, self.ACT_PER_BODY) * self._max_torque[:, None]
        zeros8 = torch.zeros((E, 8), dtype=torch.float64, device=torque.device)
        zeros1 = torch.zeros((E, 1), dtype=torch.float64, device=torque.device)
        return torch.cat([zeros8, torque, zeros1], dim=1)   # [E, U_DIM]

    def observation(self, x: torch.Tensor) -> torch.Tensor:
        quat = x[:, 7:11]
        # Canonicalize to the q_w >= 0 hemisphere (double cover), like the
        # kernel task: sign flip is differentiable almost everywhere.
        sign = torch.where(quat[:, 3:4] < 0.0, -1.0, 1.0)
        omega = torch.einsum("eij,ej->ei", self._inertia_inv, x[:, 11:14])
        obs = torch.cat([quat * sign, omega, x[:, 14:18]], dim=1)   # [E, 11]
        return obs.reshape(self._num_envs, self.obs_dim)

    def reward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        quat = x[:, 7:11]
        qn = quat / quat.norm(dim=1, keepdim=True)
        qv_sq = (qn[:, 0:3] ** 2).sum(dim=1)
        angle = 2.0 * torch.atan2(torch.sqrt(qv_sq + _ANGLE_EPS), qn[:, 3].abs())

        omega = torch.einsum("eij,ej->ei", self._inertia_inv, x[:, 11:14])
        rate_sq = (omega ** 2).sum(dim=1)

        per_entity = self.w_att * angle ** 2 + self.w_rate * rate_sq
        per_env = per_entity.reshape(self._num_envs, self._num_bodies).sum(dim=1)
        ctrl_sq = (action ** 2).sum(dim=1)
        return -(per_env + self.w_ctrl * ctrl_sq)


def default_shac_cfg(
    actor_hidden=(128, 64),
    critic_hidden=(128, 128),
    actor_lr: float = 2.0e-3,
    critic_lr: float = 5.0e-4,
) -> dict:
    """SHAC defaults close to the published SHAC settings. A convenience
    baseline, not a tuned recipe — tweak or replace freely."""
    return {
        "gamma": 0.99,
        "lam": 0.95,                    # TD(lambda)
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "lr_schedule": "linear",        # decay to ~0 over train(num_updates)
        "betas": (0.7, 0.95),           # Adam betas (as published for SHAC)
        "target_alpha": 0.995,          # Polyak factor for the target critic
        "critic_epochs": 16,
        "critic_minibatches": 4,
        "max_grad_norm": 1.0,
        "actor_hidden": list(actor_hidden),
        "critic_hidden": list(critic_hidden),
        "init_log_std": -1.0,
        "obs_normalization": True,
        # Seeds torch (network init, exploration noise, minibatch shuffling);
        # warp-side reset randomness is seeded by the simulator's reset_config.
        "seed": None,
    }


def _mlp(in_dim: int, out_dim: int, hidden) -> nn.Sequential:
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ELU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """tanh-squashed Gaussian policy with learnable per-dim log-std."""

    def __init__(self, obs_dim: int, act_dim: int, hidden, init_log_std: float):
        super().__init__()
        self.net = _mlp(obs_dim, act_dim, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mu = self.net(obs)
        if deterministic:
            return torch.tanh(mu)
        # Reparameterized sample: the analytic gradient flows through it.
        return torch.tanh(mu + torch.exp(self.log_std) * torch.randn_like(mu))


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden):
        super().__init__()
        self.net = _mlp(obs_dim, 1, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class RunningMeanStd:
    """Running observation mean/std (float64 stats, parallel Welford)."""

    def __init__(self, dim: int, device, eps: float = 1e-5):
        self.mean = torch.zeros(dim, dtype=torch.float64, device=device)
        self.var = torch.ones(dim, dtype=torch.float64, device=device)
        self.count = 0.0
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).double()
        b_mean, b_var, b_count = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean += delta * (b_count / total)
        self.var = (self.count * self.var + b_count * b_var
                    + delta ** 2 * self.count * b_count / total) / total
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # mean/var are constants on the graph; safe on the actor-loss path.
        return (x - self.mean) / torch.sqrt(self.var + self.eps)


def td_lambda_targets(
    rewards: torch.Tensor,   # [H, N]
    values: torch.Tensor,    # [H+1, N] (target-critic values, detached)
    gamma: float,
    lam: float,
) -> torch.Tensor:           # [H, N]
    """TD(lambda) targets, always bootstrapping (truncation-only episodes)."""
    H = rewards.shape[0]
    targets = torch.empty_like(rewards)
    G = values[H]
    for t in reversed(range(H)):
        G = rewards[t] + gamma * ((1.0 - lam) * values[t + 1] + lam * G)
        targets[t] = G
    return targets


class SHAC:
    """Short-Horizon Actor-Critic trainer over a `DiffVecEnv`.

    Args:
        env: The differentiable environment (owns the horizon and episode
            time limit).
        cfg: Overrides merged onto `default_shac_cfg`.
    """

    def __init__(self, env: DiffVecEnv, cfg: dict = None):
        self.env = env
        self.cfg = {**default_shac_cfg(), **(cfg or {})}
        c = self.cfg
        # Each window gives horizon * num_envs critic samples; more
        # minibatches than that leaves some empty (NaN loss).
        samples = env.horizon * env.num_envs
        m = c["critic_minibatches"]
        if isinstance(m, bool) or not isinstance(m, int) or m < 1:
            raise ValueError(f"critic_minibatches must be a positive integer, got {m!r}")
        if m > samples:
            raise ValueError(
                f"critic_minibatches ({c['critic_minibatches']}) exceeds the "
                f"{samples} critic samples per window (horizon * num_envs)"
            )
        self.device = env.device
        if c["seed"] is not None:
            torch.manual_seed(c["seed"])

        self.actor = Actor(env.obs_dim, env.act_dim, c["actor_hidden"],
                           c["init_log_std"]).to(self.device)
        self.critic = Critic(env.obs_dim, c["critic_hidden"]).to(self.device)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=c["actor_lr"], betas=c["betas"])
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=c["critic_lr"], betas=c["betas"])

        self.obs_rms = (RunningMeanStd(env.obs_dim, self.device)
                        if c["obs_normalization"] else None)
        self.update_count = 0   # completed train_step calls (drives lr decay)
        env.reset()

    # ------------------------------------------------------------- policy

    def _norm(self, obs: torch.Tensor) -> torch.Tensor:
        return self.obs_rms.normalize(obs) if self.obs_rms is not None else obs

    def policy(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Observation `[N, obs_dim]` f64 -> action `[N, act_dim]` f64.

        The single f64 -> f32 -> f64 cast point between dynamics and networks.
        """
        return self.actor(self._norm(obs).float(), deterministic).double()

    # ------------------------------------------------------- checkpointing

    def save(self, path) -> None:
        """Checkpoint networks, optimizers and observation statistics."""
        state = {
            "cfg": self.cfg,
            "update_count": self.update_count,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }
        if self.obs_rms is not None:
            state["obs_rms"] = {"mean": self.obs_rms.mean, "var": self.obs_rms.var,
                                "count": self.obs_rms.count}
        torch.save(state, path)

    def load(self, path) -> None:
        """Restore a checkpoint saved with `save`.

        The trainer must be built with the same observation/action dimensions
        and network sizes as the one that saved it.
        """
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.update_count = state.get("update_count", 0)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        if self.obs_rms is not None and "obs_rms" in state:
            self.obs_rms.mean = state["obs_rms"]["mean"].to(self.device)
            self.obs_rms.var = state["obs_rms"]["var"].to(self.device)
            self.obs_rms.count = state["obs_rms"]["count"]

    # ----------------------------------------------------------- training

    def train_step(self) -> dict:
        """One SHAC update: window rollout, actor step, critic regression,
        target update, window advance. Returns scalar stats."""
        c, env = self.cfg, self.env
        H, N = env.horizon, env.num_envs

        window = env.rollout(self.policy)
        rewards = torch.stack(window.rewards)                     # [H, N] f64, on-graph

        # Actor: analytic gradient through dynamics + policy + terminal value.
        discounts = c["gamma"] ** torch.arange(
            H, dtype=torch.float64, device=self.device)[:, None]  # [H, 1]
        v_terminal = self.target_critic(self._norm(window.obs[-1]).float())
        returns = (discounts * rewards).sum(0) + c["gamma"] ** H * v_terminal
        actor_loss = -returns.mean() / H

        self.actor_opt.zero_grad()
        actor_loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), c["max_grad_norm"])
        self.actor_opt.step()

        # Critic: TD(lambda) regression on the same window (all detached).
        with torch.no_grad():
            obs = torch.stack([o.detach() for o in window.obs])   # [H+1, N, obs]
            obs_n = self._norm(obs).float()
            values = self.target_critic(obs_n.reshape(-1, env.obs_dim)).reshape(H + 1, N)
            targets = td_lambda_targets(rewards.detach().float(), values,
                                        c["gamma"], c["lam"])
        flat_obs = obs_n[:-1].reshape(H * N, env.obs_dim)
        flat_targets = targets.reshape(H * N)

        # Accumulated on device: one host sync in the stats dict, not one
        # per minibatch.
        critic_loss_total = torch.zeros((), device=self.device)
        mb_size = (H * N) // c["critic_minibatches"]
        for _ in range(c["critic_epochs"]):
            perm = torch.randperm(H * N, device=self.device)
            for k in range(c["critic_minibatches"]):
                idx = perm[k * mb_size:(k + 1) * mb_size]
                loss = nn.functional.mse_loss(self.critic(flat_obs[idx]), flat_targets[idx])
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), c["max_grad_norm"])
                self.critic_opt.step()
                critic_loss_total += loss.detach()

        # Polyak-averaged target critic.
        with torch.no_grad():
            a = c["target_alpha"]
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.mul_(a).add_(p, alpha=1.0 - a)

        if self.obs_rms is not None:
            self.obs_rms.update(obs)   # after use: stats stay fixed per window

        env.end_window()
        self.update_count += 1
        return {
            "mean_reward": float(rewards.detach().mean()),
            "actor_loss": float(actor_loss.detach()),
            "actor_grad_norm": float(actor_grad),
            "critic_loss": float(critic_loss_total)
                / (c["critic_epochs"] * c["critic_minibatches"]),
            "actor_std": float(self.actor.log_std.detach().exp().mean()),
        }

    def train(self, num_updates: int,
              log_fn: Optional[Callable[[int, dict], None]] = None) -> list:
        """Run `num_updates` SHAC updates; returns the per-update stats.

        The linear lr schedule decays over all updates done so far plus this
        call (`update_count` is checkpointed): a fresh trainer behaves as
        before, and a resumed one asked for the remaining updates continues
        the original decay instead of restarting at full lr.

        Args:
            num_updates: Number of SHAC updates (windows) to run in this call.
            log_fn: Optional callable `(update_index, stats_dict)` for
                progress reporting.
        """
        base_alr, base_clr = self.cfg["actor_lr"], self.cfg["critic_lr"]
        total = self.update_count + num_updates
        history = []
        for _ in range(num_updates):
            if self.cfg["lr_schedule"] == "linear":
                frac = 1.0 - self.update_count / total
                for group in self.actor_opt.param_groups:
                    group["lr"] = base_alr * frac
                for group in self.critic_opt.param_groups:
                    group["lr"] = base_clr * frac
            stats = self.train_step()
            history.append(stats)
            if log_fn is not None:
                log_fn(self.update_count - 1, stats)
        return history
