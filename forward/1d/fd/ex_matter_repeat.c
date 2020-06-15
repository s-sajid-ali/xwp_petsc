
static char help[] ="X-ray propagation in matter in 1D\n\
Solves a simple time-independent linear PDE .\n\
Input parameters include:\n\
-m <points>, where <points> = number of grid points\n\
-debug              : Activate debugging printouts\n\n";

/* ------------------------------------------------------------------------

   This program solves the one-dimensional helmholtz equation
   u_t = A*u_xx + F_t*u,

   We discretize the right-hand side using finite differences with
   uniform grid spacing h:
   u_xx = (u_{i+1} - 2u_{i} + u_{i-1})/(h^2)

   ------------------------------------------------------------------------- */

#include <petscts.h>
#include <petscdraw.h>
#include <petscviewerhdf5.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
   */
typedef struct {
  PetscInt    m;                 /* total number of grid points */
  PetscReal   step_grid;         /* grid spacing */
  PetscReal   step_time;         /* step size in time */
  PetscReal   slices;            /* number of slices through object */
  PetscReal   lambda;            /* wavelength */
  PetscBool   debug;             /* flag (1 indicates activation of debugging printouts) */
  Mat         A;                 /* RHS mat*/
  Vec         slice_rid;         /* vector to hold the refractive index */
  PetscViewer hdf5_sol_viewer;   /* viewer to write the solution to hdf5*/
} AppCtx;

/*
   User-defined routines
   */
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode RHSMatrixFreeSpace(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);

int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  TS             ts;                     /* timestepping context */
  Mat            A;                      /* matrix data structure */
  Vec            u;                      /* approximate solution vector */
  PetscReal      prop_distance;          /* propagation distance */
  PetscInt prop_steps; /* number of steps for propagation */
  PetscErrorCode ierr;
  PetscInt       steps,m;
  PetscMPIInt    size,rank;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  m    = 25000;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-debug",&appctx.debug);CHKERRQ(ierr);

  appctx.m         = m;
  appctx.step_grid = 2e-9;
  appctx.lambda    = 1.23984e-10;


  prop_distance   = 6e-6;
  ierr = PetscOptionsGetReal(NULL,NULL,"-prop_distance",&prop_distance,NULL);CHKERRQ(ierr);

  prop_steps      = 4;
  ierr = PetscOptionsGetInt(NULL,NULL,"-prop_steps",&prop_steps,NULL);CHKERRQ(ierr);

  appctx.step_time = prop_distance/prop_steps;

  if(rank==0){
    ierr = PetscPrintf(PETSC_COMM_SELF,"Solving a linear TS problem on 1 processor\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"m : %d, slices : %f, lambda : %e\n",appctx.m, appctx.slices, appctx.lambda);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create a vector to hold refractive index at appctx->slice_rid
     Destroy the viewer
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscViewer hdf_ref_index_viewer;
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"grating_ref_index.h5",
      FILE_MODE_READ,&hdf_ref_index_viewer);
  CHKERRQ(ierr);

  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,
      appctx.m,&appctx.slice_rid);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) appctx.slice_rid,"ref_index");CHKERRQ(ierr);

  ierr = VecLoad(appctx.slice_rid,hdf_ref_index_viewer);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&hdf_ref_index_viewer);CHKERRQ(ierr);

  /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create vector data structures for approximate solution
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,m,&u); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u, "sol_vec");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Store solution as hdf5
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"grating_solution.h5",
      FILE_MODE_WRITE,&appctx.hdf5_sol_viewer);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set optional user-defined monitoring routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSMonitorSet(ts,Monitor,&appctx,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

     Create matrix data structure; set matrix evaluation routine.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,m);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  appctx.A = NULL;
  /*
     For linear problems with a time-dependent f(u,t) in the equation
     u_t = f(u,t), the user provides the discretized right-hand-side
     as a time-dependent matrix.
     */
  ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,RHSMatrixFreeSpace,&appctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solution vector and initial timestep
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSSetTimeStep(ts,appctx.step_time);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize timestepping solver:
     - Set the solution method to be the Backward Euler method.
     - Set timestepping duration info
     Then set runtime options, which can override these defaults.
     For example,
     -ts_max_steps <maxsteps> -ts_final_time <maxtime>
     to override the defaults set by TSSetMaxSteps()/TSSetMaxTime().
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSSetMaxSteps(ts,prop_steps);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,prop_distance);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Evaluate initial conditions
     */
  ierr = InitialConditions(u,&appctx);CHKERRQ(ierr);

  /*
     Run the timestepping solver
     */
  ierr = TSSolve(ts,u);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     View timestepping solver info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSView(ts,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.A);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.slice_rid);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&appctx.hdf5_sol_viewer);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
     - finalizes the PETSc libraries as well as MPI
     - provides summary and diagnostic information if certain runtime
     options are chosen (e.g., -log_view).
     */
  ierr = PetscFinalize();
  return ierr;
}
/* --------------------------------------------------------------------- */
/*
   InitialConditions - Computes the solution at the initial time.

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
   */
PetscErrorCode InitialConditions(Vec u,AppCtx *appctx)
{
  PetscErrorCode ierr;
  PetscInt       i,low,high;
  PetscScalar    val;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Get local vector storage info, set values and assemble
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = VecGetOwnershipRange(u,&low,&high); CHKERRQ(ierr);

  for (i=low; i<high; i++) {
    val = 1;
    ierr = VecSetValues(u,1,&i,&val,INSERT_VALUES);
  }

  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);

  /*
     Print debugging information if desired
     */
  if (appctx->debug) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Initial guess vector\n");CHKERRQ(ierr);
    ierr = VecView(u,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  return 0;
}

/* --------------------------------------------------------------------- */
/*
   Monitor - User-provided routine to monitor the solution computed at
   each timestep.  This example plots the solution and computes the
   error in two different norms.

   This example also demonstrates changing the timestep via TSSetTimeStep().

   Input Parameters:
   ts     - the timestep context
   step   - the count of the current step (with 0 meaning the
   initial condition)
   time   - the current time
   u      - the solution at this timestep
   ctx    - the user-provided context for this monitoring routine.
   In this case we use the application context which contains
   information about the problem size, workspace and the exact
   solution.
   */
PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal time,Vec u,void *ctx)
{
  AppCtx         *appctx = (AppCtx*) ctx;   /* user-defined application context */
  PetscErrorCode ierr;
  PetscInt iteration_number = time/appctx->step_time;
  ierr = PetscViewerHDF5SetTimestep(appctx->hdf5_sol_viewer,
      iteration_number);CHKERRQ(ierr);
  ierr = VecView(u,appctx->hdf5_sol_viewer);CHKERRQ(ierr);

  /*- - - - - - - - - - - - - - - - - - - -
    Print debugging information if desired
    - - - - - - - - - - - - - - - - - - - -*/
  if (appctx->debug) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Computed solution vector\n");CHKERRQ(ierr);
    ierr = VecView(u,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  return 0;
}
/* --------------------------------------------------------------------- */
/*
   RHSMatrixFreeSpace - User-provided routine to compute the right-hand-side
   matrix for the heat equation.

   Input Parameters:
   ts - the TS context
   t - current time
   global_in - global input vector
   dummy - optional user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
   AA - Jacobian matrix
   BB - optionally different preconditioning matrix
   str - flag indicating matrix structure

Notes:
Recall that MatSetValues() uses 0-based row and column numbers
in Fortran as well as in C.
*/
PetscErrorCode RHSMatrixFreeSpace(TS ts,PetscReal t,Vec X,Mat AA,Mat BB,void *ctx)
{
  Mat            A       = AA;                /* Jacobian matrix */
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscInt       low,high;
  PetscErrorCode ierr;
  PetscInt       i,idx[3];
  PetscScalar    v[3];
  PetscComplex   prefac = (-1*PETSC_i*appctx->lambda/(4*PETSC_PI))*(1/(appctx->step_grid*appctx->step_grid));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute entries for the locally owned part of the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatGetOwnershipRange(A,&low,&high); CHKERRQ(ierr);

  /*
     Set matrix rows corresponding to interior data.  We construct the
     matrix one row at a time.
     */


  for (i=low; i<high; i++) {
    //Set matrix rows corresponding to boundary data
    if (i == 0){
      v[0] = 1.0;
      ierr = MatSetValues(A,1,&i,1,&i,v,INSERT_VALUES);CHKERRQ(ierr);
    }
    else if (i == appctx->m-1){
      v[0] = 1.0;
      ierr = MatSetValues(A,1,&i,1,&i,v,INSERT_VALUES);CHKERRQ(ierr);
    }
    else{
      v[0] = prefac*-1; v[1] = prefac*2; v[2] = prefac*-1;
      idx[0] = i-1; idx[1] = i; idx[2] = i+1;
      ierr   = MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Complete the matrix assembly process and set some options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Assemble matrix, using the 2-step process:
     MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
     */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);




  ierr = MatDiagonalSet(A,appctx->slice_rid,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
     Set and option to indicate that we will never add a new nonzero location
     to the matrix. If we do, it will generate an error.
     */
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  return 0;
}

