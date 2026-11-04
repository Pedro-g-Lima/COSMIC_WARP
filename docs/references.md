# References

Sources of the models, equations and constants on the theory pages, and of
the software COSMIC WARP builds on.

## Gravity, geopotential and astrodynamics

<a id="ref-1"></a>
**[1]** Pavlis, N. K., Holmes, S. A., Kenyon, S. C., & Factor, J. K. (2012).
The development and evaluation of the Earth Gravitational Model 2008
(EGM2008). *Journal of Geophysical Research: Solid Earth*, 117(B4), B04406.
[doi:10.1029/2011JB008916](https://doi.org/10.1029/2011JB008916)
— source of the default Earth constants $\mu_\oplus$, $R_\oplus$ and $J_2$.

<a id="ref-2"></a>
**[2]** Vallado, D. A. (2013). *Fundamentals of Astrodynamics and
Applications* (4th ed.). Microcosm Press / Springer. ISBN 978-1881883180.
— Cartesian $J_2$ perturbation and its secular rates, orbital-element
conversions, the piecewise-exponential atmosphere (Table 8-4) and the
sidereal rotation rate $\omega_\oplus = 7.2921159 \times 10^{-5}\ \mathrm{rad/s}$.

<a id="ref-3"></a>
**[3]** Montenbruck, O., & Gill, E. (2000). *Satellite Orbits: Models,
Methods and Applications*. Springer.
[doi:10.1007/978-3-642-58351-3](https://doi.org/10.1007/978-3-642-58351-3)
— orbital perturbation models, gravity-gradient torque and numerical
propagation.

<a id="ref-4"></a>
**[4]** Petit, G., & Luzum, B. (eds.) (2010). *IERS Conventions (2010)*.
IERS Technical Note No. 36. Verlag des Bundesamts für Kartographie und
Geodäsie, Frankfurt am Main. — Earth's nominal mean angular velocity
$\omega_\oplus = 7.292115 \times 10^{-5}\ \mathrm{rad/s}$.

## Attitude dynamics and actuators

<a id="ref-5"></a>
**[5]** Hughes, P. C. (1986). *Spacecraft Attitude Dynamics*. Wiley.
(Dover reprint, 2004.) ISBN 978-0486439259. — Euler's rotational equations
(including bodies carrying wheels) and the gravity-gradient torque.

<a id="ref-6"></a>
**[6]** Wie, B. (2008). *Space Vehicle Dynamics and Control* (2nd ed.). AIAA
Education Series. [doi:10.2514/4.860119](https://doi.org/10.2514/4.860119)
— rigid-body and reaction-wheel attitude dynamics, torque-free precession.

<a id="ref-7"></a>
**[7]** Markley, F. L., & Crassidis, J. L. (2014). *Fundamentals of
Spacecraft Attitude Determination and Control*. Springer.
[doi:10.1007/978-1-4939-0802-8](https://doi.org/10.1007/978-1-4939-0802-8)
— quaternion kinematics and reaction-wheel momentum exchange.

<a id="ref-8"></a>
**[8]** Sidi, M. J. (1997). *Spacecraft Dynamics and Control: A Practical
Engineering Approach*. Cambridge University Press.
[doi:10.1017/CBO9780511815652](https://doi.org/10.1017/CBO9780511815652)
— reaction-wheel modeling, saturation and momentum management.

<a id="ref-9"></a>
**[9]** Wertz, J. R. (ed.) (1978). *Spacecraft Attitude Determination and
Control*. Astrophysics and Space Science Library, Vol. 73. Reidel / Springer.
[doi:10.1007/978-94-009-9907-7](https://doi.org/10.1007/978-94-009-9907-7)
— centered-dipole magnetic field model and residual-dipole torque; equatorial
surface strength $B_0 \approx 3.12 \times 10^{-5}\ \mathrm{T}$.

<a id="ref-10"></a>
**[10]** Shuster, M. D. (1993). A survey of attitude representations.
*Journal of the Astronautical Sciences*, 41(4), 439–517. — quaternion
conventions and kinematics.

## Geomagnetic field

<a id="ref-11"></a>
**[11]** Alken, P., Thébault, E., Beggan, C. D., et al. (2021). International
Geomagnetic Reference Field: the thirteenth generation. *Earth, Planets and
Space*, 73, 49.
[doi:10.1186/s40623-020-01288-x](https://doi.org/10.1186/s40623-020-01288-x)
— reference geomagnetic field the centered dipole approximates.

## Numerical integration

<a id="ref-12"></a>
**[12]** Hairer, E., Lubich, C., & Wanner, G. (2006). *Geometric Numerical
Integration: Structure-Preserving Algorithms for Ordinary Differential
Equations* (2nd ed.). Springer Series in Computational Mathematics, Vol. 31.
[doi:10.1007/3-540-30666-8](https://doi.org/10.1007/3-540-30666-8)
— symplectic integrators and the Störmer–Verlet method.

<a id="ref-13"></a>
**[13]** Leimkuhler, B., & Reich, S. (2004). *Simulating Hamiltonian
Dynamics*. Cambridge University Press.
[doi:10.1017/CBO9780511614118](https://doi.org/10.1017/CBO9780511614118)
— splitting methods and kick–drift–kick schemes.

<a id="ref-14"></a>
**[14]** Butcher, J. C. (2016). *Numerical Methods for Ordinary Differential
Equations* (3rd ed.). Wiley.
[doi:10.1002/9781119121534](https://doi.org/10.1002/9781119121534)
— forward Euler and the classical fourth-order Runge–Kutta scheme.

<a id="ref-15"></a>
**[15]** Verlet, L. (1967). Computer "experiments" on classical fluids. I.
Thermodynamical properties of Lennard-Jones molecules. *Physical Review*,
159(1), 98–103.
[doi:10.1103/PhysRev.159.98](https://doi.org/10.1103/PhysRev.159.98)
— the Störmer–Verlet scheme in its velocity form.

## Propulsion

<a id="ref-16"></a>
**[16]** Sutton, G. P., & Biblarz, O. (2016). *Rocket Propulsion Elements*
(9th ed.). Wiley. ISBN 978-1118753651. — specific impulse and the
mass-flow / rocket equation ($g_0 = 9.80665\ \mathrm{m/s^2}$).

## Automatic differentiation

<a id="ref-17"></a>
**[17]** Griewank, A., & Walther, A. (2008). *Evaluating Derivatives:
Principles and Techniques of Algorithmic Differentiation* (2nd ed.). SIAM.
[doi:10.1137/1.9780898717761](https://doi.org/10.1137/1.9780898717761)
— reverse-mode automatic differentiation and the adjoint method.

## Software

<a id="ref-18"></a>
**[18]** Macklin, M. (2022). *Warp: A High-performance Python Framework for
GPU Simulation and Graphics*. NVIDIA GPU Technology Conference (GTC).
[github.com/NVIDIA/warp](https://github.com/NVIDIA/warp)
— the GPU kernel framework the simulator is built on.

<a id="ref-19"></a>
**[19]** Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020).
Array programming with NumPy. *Nature*, 585, 357–362.
[doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)
— host-side arrays, configuration checks and the orbital-element helpers.

<a id="ref-20"></a>
**[20]** Hunter, J. D. (2007). Matplotlib: A 2D graphics environment.
*Computing in Science & Engineering*, 9(3), 90–95.
[doi:10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)
— figures in the examples.

## Sampling and celestial mechanics

<a id="ref-21"></a>
**[21]** Shoemake, K. (1992). Uniform random rotations. In D. Kirk (ed.),
*Graphics Gems III*, 124–132. Academic Press.
[doi:10.1016/B978-0-08-050755-2.50036-1](https://doi.org/10.1016/B978-0-08-050755-2.50036-1)
— uniform attitude sampling on SO(3).

<a id="ref-22"></a>
**[22]** Danby, J. M. A. (1992). *Fundamentals of Celestial Mechanics*
(2nd ed.). Willmann-Bell. ISBN 978-0943396200. — Kepler-equation solver and starter used for
mean-anomaly sampling.

<a id="ref-23"></a>
**[23]** Salmon, J. K., Moraes, M. A., Dror, R. O., Shaw, D. E. (2011).
Parallel random numbers: as easy as 1, 2, 3. *Proceedings of the
International Conference for High Performance Computing, Networking, Storage
and Analysis (SC '11)*, 16:1–16:12.
[doi:10.1145/2063384.2063405](https://doi.org/10.1145/2063384.2063405)
— the Philox4x32-10 counter-based generator behind resets and randomization.

## BibTeX

All entries above are available as a BibTeX file:
[`references.bib`](references.bib).
