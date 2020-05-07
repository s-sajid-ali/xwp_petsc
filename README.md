## xwp_petsc : X-ray wave propagation in PETSc
Code for evaluation of wave propagation (fourier and real space approaches) using PETSc. Works for 1D and 2D for free space and matter.

#### Currently implemented  :
- MPI - Multislice : Iterate between slice projections diffraction and free space propagation
- Finite Difference method : Solve the Helmholtz PDE using the TS integrator and Multigrid preconditioning.

#### In Progress :
- Tomography simulation by rotating the object using a rotation matrix.

The phantom test case conatins data from [EMD-3756](https://www.emdataresource.org/EMD-3756)
