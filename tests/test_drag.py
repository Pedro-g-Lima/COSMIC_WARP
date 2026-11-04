"""
Atmospheric drag validation.

Density is checked against the host-side copy of the Vallado Table 8-4
piecewise-exponential model. Orbital decay is checked against the analytic
secular rate for a circular orbit in a co-rotating atmosphere:

    da/dt = -rho * (Cd*A/m) * v_rel^2 * sqrt(a^3/mu)

with v_rel = v_c - omega_earth * a for an equatorial prograde orbit.
"""
import numpy as np
import warp as wp

from cosmic_warp.physics.drag import atmosphere_density, DENSITY_TABLE, OMEGA_EARTH
from cosmic_warp.physics.gravity import MU_EARTH, RE_EARTH
from conftest import set_state
from orbit_utils import circular_velocity


@wp.kernel
def _density_kernel(radii: wp.array(dtype=wp.float64),
                    densities: wp.array(dtype=wp.float64)):
    tid = wp.tid()
    pos = wp.vec3d(radii[tid], wp.float64(0.0), wp.float64(0.0))
    densities[tid] = atmosphere_density(pos)


def _reference_density(h: float) -> float:
    """Host-side oracle: evaluate the piecewise-exponential table directly."""
    h = max(h, 0.0)
    h0, rho0, hs = DENSITY_TABLE[0]
    for band in DENSITY_TABLE:
        if h >= band[0]:
            h0, rho0, hs = band
    return rho0 * np.exp(-(h - h0) / hs)


def test_density_profile(device):
    """Kernel density must match the reference table in every band."""
    # Mid-band altitudes for all 28 bands, plus edge cases
    altitudes = [0.5 * (lo[0] + hi[0]) for lo, hi in
                 zip(DENSITY_TABLE[:-1], DENSITY_TABLE[1:])]
    altitudes += [-1.0e3, 0.0, 400.0e3, 1000.0e3, 1500.0e3]

    radii = wp.array(np.asarray(altitudes) + RE_EARTH, dtype=wp.float64, device=device)
    densities = wp.zeros(len(altitudes), dtype=wp.float64, device=device)
    wp.launch(_density_kernel, dim=len(altitudes), inputs=[radii, densities], device=device)

    computed = densities.numpy()
    expected = np.array([_reference_density(h) for h in altitudes])
    np.testing.assert_allclose(computed, expected, rtol=1e-12)


def _fit_sma_rate(result, dt):
    """Fit the secular semi-major-axis rate [m/s] from saved states (env 0)."""
    pos = result["positions"].numpy()[:, 0, :]
    vel = result["velocities"].numpy()[:, 0, :]
    r = np.linalg.norm(pos, axis=1)
    v = np.linalg.norm(vel, axis=1)
    a = 1.0 / (2.0 / r - v**2 / MU_EARTH)  # vis-viva
    t = (np.arange(len(a)) + 1) * dt
    return np.polyfit(t, a, 1)[0], a


def test_orbital_decay_rate(make_sim):
    a = RE_EARTH + 400.0e3  # circular LEO at 400 km
    r0 = np.array([a, 0.0, 0.0])
    v_c = circular_velocity(MU_EARTH, a)
    v0 = np.array([0.0, v_c, 0.0])  # equatorial prograde

    dt = 5.0
    num_steps = 2200  # ~2 orbits (small altitude loss keeps density ~constant)
    sim = make_sim(num_envs=1, integrator="symplectic", dt=dt, drag=True)
    set_state(sim, pos=[r0], vel=[v0])

    result = sim.simulate(num_steps=num_steps, save_states=True)
    fitted_rate, _ = _fit_sma_rate(result, dt)

    # Analytic secular decay with default vehicle (m=1 kg, A=1 m^2, Cd=2.2)
    rho = _reference_density(400.0e3)
    v_rel = v_c - OMEGA_EARTH * a  # co-rotating atmosphere, equatorial orbit
    analytic_rate = -rho * 2.2 * v_rel**2 * np.sqrt(a**3 / MU_EARTH)

    rel_err = abs(fitted_rate - analytic_rate) / abs(analytic_rate)
    assert rel_err < 0.10, (
        f"da/dt {fitted_rate:.3e} m/s vs analytic {analytic_rate:.3e} m/s "
        f"(rel err {rel_err:.1%})"
    )


def test_drag_flag_off_no_decay(make_sim):
    """With drag disabled the semi-major axis must not decay."""
    a = RE_EARTH + 400.0e3
    r0 = np.array([a, 0.0, 0.0])
    v0 = np.array([0.0, circular_velocity(MU_EARTH, a), 0.0])

    dt = 5.0
    sim = make_sim(num_envs=1, integrator="symplectic", dt=dt, drag=False)
    set_state(sim, pos=[r0], vel=[v0])

    result = sim.simulate(num_steps=2200, save_states=True)
    fitted_rate, _ = _fit_sma_rate(result, dt)

    # Drag at 400 km decays a at ~0.4 m/s; without drag the fitted rate is
    # round-off (measured 4.4e-9 m/s)
    assert abs(fitted_rate) < 1e-7, (
        f"semi-major axis drifted without drag: da/dt = {fitted_rate:.3e} m/s"
    )
