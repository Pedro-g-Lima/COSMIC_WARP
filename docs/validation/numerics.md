# Numerics

Background: [numerical integration](../theory/integrators.md).

## Order of convergence

`tests/test_two_body.py`, `tests/test_integrator_order.py`: the observed
order is log₂ of the error ratio per halving of dt; every ratio must clear
the bound ([[12]](../references.md#ref-12), [[14]](../references.md#ref-14)).
Against the Kepler solution (2048 s, a = 7500 km, e = 0.1):

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Euler, dt = 2, 1, 0.5 s | analytic solution; expected order 1 | observed > 0.5 |
| Symplectic, dt = 8, 4, 2 s | analytic solution; expected order 2 | observed > 1.5 |
| RK4, dt = 32, 16, 8 s | analytic solution; expected order 4 | observed > 3.5 |

![Position error against the time step for Euler, symplectic and RK4 on the Kepler orbit, log-log, with slopes 1, 2 and 4](../figures/convergence-light.svg#only-light)
![Position error against the time step for Euler, symplectic and RK4 on the Kepler orbit, log-log, with slopes 1, 2 and 4](../figures/convergence-dark.svg#only-dark)

/// caption
Position error after 2048 s at the tested step sizes; dashed lines have the
expected slopes. Data: `tests/test_two_body.py` (`test_convergence_order`),
computed on CPU at site build.
///

Elsewhere, self-convergence: with y(n) the full final state after n steps
over a fixed span, e(n) = |y(n) − y(2n)| must shrink by 2^order per halving;
all three integrators per scenario.

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Asymmetric tumble (I = diag(1, 2, 3), non-principal spin), 4 s | self-convergence; expected 1 / 2 / 4 | observed > expected − 0.3 |
| Damped wheels on three axes with a transverse body rate, so ω × (L + h) is active, 4 s | self-convergence; expected 1 / 2 / 4 | observed > expected − 0.3 |
| Finite burn in orbit: thrust, mass flow and gravity together, 100 s | self-convergence; expected 1 / 2 / 4 | observed > expected − 0.3 |
| A wheel saturating at t = 1.35 s, inside a step for every n | self-convergence; expected order capped at 2 (`ORDER_CAP`): 1 / 2 / 2 | observed > expected − 0.3 |

The cap is a property of the method, not a tolerance: no integrator locates
the saturation clamp, a kink in the dynamics. Burnout:
[thrusters](physics.md#thrusters-and-the-rocket-equation).

## Integrator agreement

`tests/test_integrator_order.py`: self-convergence checks the rate, not the
answer, so this compares the answers (the symplectic kick and drift share no
stepping code with Euler and RK4).

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Each integrator's distance from a fine RK4 reference (4× the finest RK4 step count), in the tumble, wheel-exchange and burn scenarios | its own Richardson estimate \|y(2n) − y(n)\| / (2^order − 1) | ratio in 0.8–1.2 |
| Same, wheel-saturation scenario | as above, with the capped order | ratio in 0.3–2.0 (measured 0.75 for RK4) |

At a mid-step kink the error constant changes as the step halves, so the
estimate only gives the scale.

## Conserved quantities

`tests/test_two_body.py`, `tests/test_attitude.py`: the largest deviation
along the trajectory ([[12]](../references.md#ref-12),
[[13]](../references.md#ref-13)). One orbit (a = 7500 km, e = 0.1) at
dt = 1 s, sampled every 10 s:

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Specific energy, symplectic | invariant | 1e-6 relative (measured 1.0e-7) |
| Specific energy, RK4 | invariant | 1e-13 relative |
| Angular-momentum vector, both | invariant | 1e-13 relative (measured ≤ 8e-15) |
| Eccentricity magnitude, symplectic / RK4 | invariant | 1e-5 (measured 5.0e-7) / 1e-12 |
| Eccentricity (Laplace–Runge–Lenz) vector, relative to e, symplectic / RK4 | invariant | 1e-4 (measured 5.1e-6) / 1e-11 |

Torque-free tumble of an asymmetric body (I = diag(0.2, 0.3, 0.4)), 5 s at
dt = 1 ms:

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Inertial angular momentum, symplectic / RK4 | invariant | 1e-12 (measured 3.3e-14) / 1e-13 (measured 5.7e-15) relative |
| Rotational kinetic energy, symplectic / RK4 | invariant | 1e-7 (measured 1.2e-9) / 1e-13 (measured 7.6e-15) relative |
| Quaternion norm, both | unit norm | 1e-12 |

With reaction wheels: [reaction wheels](physics.md#reaction-wheels).
Long-horizon behaviour is not tested.

## Golden trajectories

`tests/test_golden.py`

| Claim | Referent | Tolerance |
| --- | --- | --- |
| 150 steps of a seeded two-body scenario per integrator — every physics model, thrusters and wheels, termination with automatic resets and domain randomization — reproduce the final state, wheel momenta, termination status, episode counters, RNG streams and randomized parameters | a committed per-device fixture (`tests/goldens/`) | bitwise |
| The scenario resets environments during the run | episode counters | exact |

They catch internally consistent changes property tests miss (a Warp
release altering codegen or the random streams, an accidental physics edit).
Fixtures are per device and bit-exact only with the Warp version the `dev`
group pins; the dependency-range CI job deselects them.
