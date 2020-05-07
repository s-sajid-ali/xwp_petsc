#include <petscmat.h>
#include <petscdmda.h>
#include <petscviewerhdf5.h>

/* User-defined application context - contains data needed
   by the application-provided call-back routines. */

#ifndef STRUCT_APPCTX
#define STRUCT_APPCTX

typedef struct {
	PetscReal   L_x;               /* Length along x */
	PetscReal   L_y;               /* Length along y */
	PetscReal   lambda;            /* wavelength */
	PetscReal   energy;            /* energy in ev */
	PetscReal   step_time;         /* step size in time */
	PetscInt    prop_steps;        /* number of steps for propagation */
	PetscViewer hdf5_sol_viewer;   /* viewer to write the solution to hdf5*/
	PetscViewer hdf5_rid_viewer;   /* viewer to read refractive index*/
	Mat         slice_rid;         /* matrix to hold the refractive index */
	Mat         slice_adj;         /* matrix to hold adjoint with respect to refractive index */
	Vec         base_diag;         /* vector to hold the diag at t=0 */
	DM          da;                /* Use DMDA to manage grid and vecs*/
	} AppCtx;

#endif

