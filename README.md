## xwp_petsc : X-ray wave propagation in PETSc
Code for evaluation of wave propagation (fourier and real space approaches) using PETSc. Works for 1D and 2D for free space and matter.

#### Currently implemented  :
- Full Array Fresnel Multislice : Iterate between slice projections diffraction and free space propagation
- Finite Difference method : Solve the Helmholtz PDE using TS integrator

#### Publications:
*Comparison of distributed memory algorithms for X-ray wave propagation in inhomogeneous media* Sajid Ali, Ming Du, Mark F. Adams, Barry Smith, and Chris Jacobsen  [Optics Express Vol. 28, Issue 20 (2020)](https://doi.org/10.1364/OE.400240) : The implementation used in this paper can be found by checking out [this release](https://github.com/s-sajid-ali/xwp_petsc/releases/tag/1.0). Links to further information on [generating the zone plate test object](https://github.com/s-sajid-ali/xwp_petsc/wiki/Generating-zone-plates-for-simulation) and [running the simulations](https://github.com/s-sajid-ali/xwp_petsc/wiki/Running-the-forward-models).
