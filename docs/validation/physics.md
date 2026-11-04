# Physics

The models behind every step: gravity, rigid-body rotation, environmental
torques, drag, thrusters and reaction wheels, plus the orbital-element
conversions behind initial-condition sampling. Each table names the test file
that checks it. All checks run in float64.

The analytic references in `tests/orbit_utils.py` (Kepler propagation,
orbital elements, the Clohessy–Wiltshire solution) are host-side NumPy and
import nothing from `cosmic_warp`, so a bug in the simulator cannot hide in
its own reference.

## Two-body motion

`tests/test_two_body.py`: an elliptical orbit (a = 7500 km, e = 0.1) started
at periapsis and compared with the analytic Kepler solution, found by a Newton
solve of Kepler's equation ([[2]](../references.md#ref-2)).

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Position after 3000 s (about half an orbit), RK4 at dt = 1 s | analytic Kepler solution | < 1e-5 m (measured 7.8e-7 m) |
| Same, symplectic at dt = 1 s | analytic Kepler solution | < 50 m (measured 9.6 m) |
| Same, Euler at dt = 0.05 s | analytic Kepler solution | < 5000 m (measured 3.7e3 m) |
| Angular-momentum vector, largest deviation over one orbit, RK4 and symplectic | invariant of a central force | < 1e-13 relative (measured ≤ 8e-15) |

Energy and eccentricity-vector conservation are on the numerics page, under
[conserved quantities](numerics.md#conserved-quantities); the observed order
against this solution is under [order of convergence](numerics.md#order-of-convergence).

## J2 oblateness

`tests/test_j2.py`: inclined orbits propagated for one day (symplectic,
dt = 10 s), with the secular rate taken as the slope of a linear fit, which
averages out the short-period J2 terms ([[2]](../references.md#ref-2),
[[3]](../references.md#ref-3)).

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Nodal regression, circular 500 km orbit at i = 51.6° | dΩ/dt = −(3/2) J₂ n (R⊕/p)² cos i | 3 % relative |
| Apsidal rotation, a = 7500 km, e = 0.1, i = 30° | dω/dt = (3/4) J₂ n (R⊕/p)² (5 cos² i − 1) | 3 % relative |
| With J2 off, the orbit plane does not move (20 000 s) | point-mass gravity keeps h fixed | unit-normal drift < 1e-9 |

The formulas are first-order secular theory, so the 3 % band bounds the
theory as much as the simulator.

![Change in the right ascension of the node and in the argument of periapsis over one day, simulated and first-order theory](../figures/j2-light.svg#only-light)
![Change in the right ascension of the node and in the argument of periapsis over one day, simulated and first-order theory](../figures/j2-dark.svg#only-dark)

/// caption
Change in the right ascension of the node (circular 500 km orbit, i = 51.6°)
and in the argument of periapsis (a = 7500 km, e = 0.1, i = 30°) over one
day: simulated (solid, with the short-period J2 terms) and the first-order
secular rate (dashed). The fitted slopes are what the test holds to 3 % of
theory. Data: `tests/test_j2.py`, computed on CPU when the site is built.
///

## Rigid-body rotation

`tests/test_attitude.py`, `tests/test_closed_form.py`: torque-free rotation
with gravity off ([[5]](../references.md#ref-5),
[[6]](../references.md#ref-6)).

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Axisymmetric body (I = diag(0.2, 0.2, 0.1)): phase of the transverse rate after 10 s, symplectic, dt = 1e-3 s | analytic body-frame precession ω_p = (I_s/I_t − 1) ω_s | 5e-5 rad (measured 2.5e-6) |
| Spin-axis rate and transverse-rate magnitude of the same body | constants of motion | 1e-9 and 5e-9 rad/s (measured 2.2e-11 and 1.3e-10) |
| Gyroscopic coupling: body spinning about x with a wheel holding h along z, one RK4 step | dL/dt = −ω × (L + h) = (0, ωh, 0) | 1e-4 relative, 1e-4·ωh absolute (measured 3e-5·ωh) |

Kinetic energy, inertial angular momentum and quaternion norm over an
asymmetric tumble are under [conserved quantities](numerics.md#conserved-quantities).

## Environmental torques

`tests/test_closed_form.py`: each test starts from rest, takes one RK4 step
of 1 ms and compares the angular impulse divided by dt with the analytic
torque. The step's own induced spin is an O(dt) correction, bounded
explicitly where it is visible.

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Gravity-gradient torque at r = 7000 km, body rotated 0.3 rad about z | τ_z = 3μ/r³ (I_x − I_y) sin θ cos θ ([[5]](../references.md#ref-5)) | 1e-6 relative |
| Magnetic torque at the equator and over the north pole | τ = m × B, aligned dipole B = B₀ (R/r)³ [3(m̂·r̂)r̂ − m̂] ([[9]](../references.md#ref-9)) | 1e-6 relative |

The magnetic test checks the dipole model the code implements, with its own
B₀ and reference radius; it does not compare against IGRF.

## Atmospheric drag

`tests/test_drag.py`: density from the piecewise-exponential model of
Vallado Table 8-4, and orbital decay against the secular rate for a circular
orbit in a co-rotating atmosphere ([[2]](../references.md#ref-2)).

| Claim | Referent | Tolerance |
| --- | --- | --- |
| In-kernel density at the middle of every band, and at −1, 0, 400, 1000 and 1500 km | the same table evaluated on the host | 1e-12 relative |
| Semi-major-axis decay, circular equatorial 400 km orbit, about two orbits | da/dt = −ρ (C_D A/m) v_rel² √(a³/μ), v_rel = v_c − ω⊕ a | 10 % relative |
| With drag off, no decay | zero | \|da/dt\| < 1e-7 m/s (measured 4.4e-9; decay with drag ≈ 0.4 m/s) |

The density check reads the table from the code (`DENSITY_TABLE`): it
verifies the in-kernel band lookup and exponential, not the table's values
against the published table.

## Thrusters and the rocket equation

`tests/test_maneuvers.py`: one thruster at the centre of mass along +x,
10 N at Isp = 200 s, gravity off ([[16]](../references.md#ref-16)).

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Δv after burning all propellant (m₀ = 2 kg, m_dry = 1 kg, dt = 0.01 s), symplectic and RK4 | Tsiolkovsky: Δv = Isp g₀ ln(m₀/m_f) | 1e-10 relative (measured 2.0e-12, 1.3e-12) |
| Same, Euler | Tsiolkovsky | 5e-4 relative (measured 1.8e-5) |
| Mass after depletion | dry mass | < 1e-9 kg |
| A thruster through the centre of mass adds no rotation | zero torque | \|L\| < 1e-12, quaternion w within 1e-12 of 1 |
| Burnout 37 % into a 1.7 s step: Δv | Tsiolkovsky | symplectic 1e-12, RK4 1e-9 relative (measured 6e-11); mass exactly dry |
| Position error from spreading the last propellant over the burnout step, dt = 1.7, 0.85, 0.425 s (RK4) | exactly −(T/m_dry) dt² f(1 − f)/2, f the fraction of the step the propellant lasts: second order | 5e-3 relative |
| Euler burnout step with propellant for 30 % of it | impulse T · 0.3 dt | 1e-12 relative |

![Burnout position error scaled by m_dry over T dt squared, against the burn fraction f, for three step sizes, on the closed form -f(1-f)/2](../figures/burnout-light.svg#only-light)
![Burnout position error scaled by m_dry over T dt squared, against the burn fraction f, for three step sizes, on the closed form -f(1-f)/2](../figures/burnout-dark.svg#only-dark)

/// caption
Position error left by the burnout step, scaled by m_dry/(T dt²), against the
fraction f of the step the propellant lasts, near each of the three step
sizes the test checks (filled: the tested step sizes themselves; hollow: step
sizes chosen to sweep f). All three fall on one curve, the closed form
−f(1 − f)/2: the error is second order in dt and at most T dt²/(8 m_dry).
Data: `tests/test_maneuvers.py` (`burnout_position_error`, RK4), computed on
CPU when the site is built.
///

`tests/test_closed_form.py`, `tests/test_thrusters.py`: the thruster wrench
and its configuration.

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Off-centre thruster, body rotated 90° about z, one RK4 step | F = T R(q) û and τ = d × F | force 1e-6 relative; torque 1e-6 relative, 2e-8 absolute (measured 6e-9) |
| A non-unit direction is stored normalized, same heading | unit vector | 1e-12 |
| Thrust magnitude for direction vectors scaled 0.5×, 2×, 10× | the unit-direction thrust | 1e-14 relative |
| Effective specific impulse \|F\|/(ṁ g₀), for unit and non-unit directions | the configured Isp | 1e-12 relative |
| An unused (zero-direction) slot | no force, no propellant | exactly 0; commanding all eight slots equals the one real thruster to 1e-14 |
| Default layout | six unit directions, two zero slots | 1e-12 |

## Reaction wheels

`tests/test_rw.py`: wheel torque is a pure exchange of angular momentum
between wheel and body ([[7]](../references.md#ref-7),
[[8]](../references.md#ref-8)). Gravity off, starting at rest, so the total
inertial angular momentum is zero.

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Undamped wheel momentum over 20 s, all three integrators | L_w = τt | 1e-12 N·m·s |
| Total inertial angular momentum, body plus wheel, all three integrators | conserved at zero | < 1e-10 N·m·s (wheel momentum reaches 1 N·m·s) |
| The body counter-rotates | I_z ω_z = −L_w | 1e-10 N·m·s |
| Saturation: the wheel never exceeds `max_momentum` and pins at it | the capacity | 1e-12 |
| Through and beyond saturation, total angular momentum | conserved at zero | < 1e-10 |
| After saturation the body rate is frozen | no further exchange | 1e-12 rad/s |
| Reversed torque unwinds a saturated wheel | below capacity | by more than 1e-6 |
| Damped wheel, 50 s | independent fine-step (dt/100) RK4 of J ω̇ = τ − bω | 1e-9 relative |
| Damped wheel for b from 1e-18 to 1 N·m·s, 1 s; tiny damping against the undamped wheel (computing 1 − e^(−bt/J) directly left the wheel at rest at b = 1e-18) | the closed form (τJ/b)(1 − e^(−bt/J)), evaluated with `expm1`; τt | 1e-13 relative |
| A zero spin axis leaves the wheel unused; a non-unit axis is normalized | specified semantics | exact |

![Wheel momentum over 2 s for an undamped wheel saturating at its capacity and a damped wheel, simulated against the closed forms](../figures/wheels-light.svg#only-light)
![Wheel momentum over 2 s for an undamped wheel saturating at its capacity and a damped wheel, simulated against the closed forms](../figures/wheels-dark.svg#only-dark)

/// caption
Wheel momentum over the first 2 s: an undamped wheel torqued into its
capacity, and a damped wheel approaching τJ/b. Lines are the closed forms,
circles the simulation (symplectic, every few steps). The test holds the
first to the capacity to 1e-12 and the second to an independent fine-step
RK4 integration to 1e-9. Data: `tests/test_rw.py` (`saturating_wheel`,
`damped_wheel`), computed on CPU when the site is built.
///

## Impulsive manoeuvres

`tests/test_maneuvers.py`: a Hohmann transfer from a 7000 km to a 9000 km
circular orbit, with both impulses from the vis-viva equation (symplectic,
dt = 1 s; [[2]](../references.md#ref-2)).

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Radius after half the transfer ellipse | the target radius r₂ | 50 m (measured 4.3 m) |
| Semi-major axis after circularizing | r₂ | 10 m (measured 0.57 m) |
| Eccentricity after circularizing | circular | < 2e-4 (measured 4.5e-5) |
| Radius a quarter of the final orbit later | r₂ | 3 km (measured 406 m, which is e·r₂) |

## Relative motion

`tests/test_cw.py`: a chaser 100 m radially outside a target on a circular
6878 km orbit, both as bodies of one environment (symplectic, dt = 1 s), so
the test also exercises the multi-body path ([[2]](../references.md#ref-2)).

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Relative position in the target's LVLH frame at eight epochs over one orbit | closed-form Clohessy–Wiltshire solution | 0.5 m, set by the CW linearization |
| Cross-track motion of a planar setup | zero | < 1e-6 m |

## Orbital elements

`tests/test_config.py`, `tests/test_orbit_sampling.py`: the host-side
conversions behind orbit initial conditions and orbit-mode resets
([[2]](../references.md#ref-2), [[22]](../references.md#ref-22)).

| Claim | Referent | Tolerance |
| --- | --- | --- |
| `elements_to_state` radius, speed, \|h\| and inclination | conic equation, vis-viva, √(μp), the requested inclination | 1e-6 m, 1e-9 m/s, 1e-3 m²/s, 1e-12 rad |
| Orientation conventions: node on [cos Ω, sin Ω, 0] heading north; ω = 90° on a polar orbit puts periapsis over the north pole; anomaly advances along the motion | the conventions (invariants alone pass under a sign flip) | 1e-12 |
| `state_to_elements` inverts `elements_to_state` | identity | 1e-9 · max(1, \|value\|) |
| … inclination at i = 1e-8 and π − 1e-8 (an `acos` form returned 1.49e-8 for 1e-8) | the requested inclination | 1e-15 rad |
| Radial, at-the-centre and hyperbolic states | undefined elements | raises |
| `circular_orbit_state` | r · v = 0; speed √(μ/r) | 1e-6; 1e-9 m/s |
| Initial condition given by mean anomaly (e = 0.3) | Kepler's equation recovered from the state | 1e-9 rad |
| Circular initial-condition orbit at i = 51.6° | radius, circular speed, inclination | radius and speed bitwise; inclination 1e-12 rad |

## Orbital-element sampling

`tests/test_orbit_sampling.py`: resets that draw classical elements on the
device ([sampling theory](../theory/sampling.md)).

| Claim | Referent | Tolerance |
| --- | --- | --- |
| Degenerate bounds, 16 environments | `elements_to_state` on the host | position 1e-12 relative + 1e-5 m; momentum 1e-12 relative + 1e-8 |
| Circular band, 4096 environments, sampled mass in [2, 4] kg | radius within bounds; vis-viva speed; zero radial velocity; momentum = mass × velocity | ±1 m; 1e-6 × mean speed; 1e-6 relative |
| All six elements recovered from 4096 draws | inside the bounds and uniform on them | bounds to 1e-6 · max(1, hi); normalized mean within 0.02 of 1/2, variance within 0.01 of 1/12 |
| Mean-anomaly mode, e = 0.5 | mean anomaly uniform in time | same moment bands; more than 60 % of draws on the apoapsis half |
| Propellant randomization after the orbit draw | momentum rescaled, velocity unchanged | speed within 1e-6 of circular |
| Orbit-mode resets inside `simulate()` with auto-reset | circular orbits at the drawn radius | speed √(μ/r) to 1e-6 relative |
| Perigee below the planet radius, from Python and from YAML | rejected on every reset | raises |
| An orbit spec with a `mu`, from Python or YAML (`tests/test_reset.py`, `tests/test_config.py`) | the dynamics only know Earth | raises |
