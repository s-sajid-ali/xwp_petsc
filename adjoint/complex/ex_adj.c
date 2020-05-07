
static char help[] ="X-ray propagation in matter in 2D\n\
Solves a simple time-independent linear PDE\n\
Uses DMDA. Basic approach adapted from Ed Bueler's book\n\
Input parameters include:\n\
  -mx,my         : number of grid points\n\
  -energy        : energy of x-rays in ev\n\
  -prop_distance : distance of propagation\n\
  -prop_steps    : number of propagation steps\n\
  -step_grid_x/y : pixel size in x and y\n\n";

/* ------------------------------------------------------------------------
   This program solves the two-dimensional helmholtz equation:

       u_t = A*u_xx + A*u_yy + F_t*u;
  ------------------------------------------------------------------------- */

#include <petscts.h>
#include <petscdmda.h>
#include <petscviewerhdf5.h>
#include "appctx.h"

/*
   User-defined routines
*/
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode formMatrix(DM, Mat, void*);
extern PetscErrorCode RHSMatrixMatter(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);

int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  TS             ts;                     /* timestepping context */
  Mat            A;                      /* matrix data structure */
  Vec            u;                      /* approximate solution vector */
  PetscReal      prop_distance;          /* propagation distance */

  PetscInt       steps;                  /* output for TSGetStepNumber */
  PetscInt       mx,my;                  /* grid size in x and y */
  PetscMPIInt    size,rank;              /* MPI size and rank*/
  PetscBool      ts_jac_reuse = true;    /* for TSRHSJacobianSetReuse*/
  PetscErrorCode ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  mx        = 4096;
  my        = 4096;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&mx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-my",&my,NULL);CHKERRQ(ierr);

  appctx.energy    = 25000;
  ierr = PetscOptionsGetReal(NULL,NULL,"-energy",&appctx.energy,NULL);
    CHKERRQ(ierr);
  appctx.lambda    = (1239.84/appctx.energy)*1e-9;

  prop_distance   = 1e-3;
  ierr = PetscOptionsGetReal(NULL,NULL,"-prop_distance",&prop_distance,NULL);CHKERRQ(ierr);

  appctx.prop_steps      = 10;
  ierr = PetscOptionsGetInt(NULL,NULL,"-prop_steps",&appctx.prop_steps,NULL);CHKERRQ(ierr);

  appctx.L_x = 10e-9 * mx;
  appctx.L_y = 10e-9 * my;
  ierr = PetscOptionsGetReal(NULL,NULL,"-L_x",
                             &appctx.L_x,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-L_y",
                             &appctx.L_y,NULL);CHKERRQ(ierr);

  appctx.step_time   = prop_distance/appctx.prop_steps;

  if(rank==0){
      ierr = PetscPrintf(PETSC_COMM_SELF,
                         "Solving a linear TS problem on %d processors\n",size);
      CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"mx : %d, my: %d, energy(in eV) : %e\n",
                         mx, my, appctx.energy);CHKERRQ(ierr);
      }



  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create DMDA object.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/

  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                      DMDA_STENCIL_STAR,mx,my,
                      PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,
                      &appctx.da);CHKERRQ(ierr);

  ierr = DMSetFromOptions(appctx.da); CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da); CHKERRQ(ierr);

 /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create a vector to hold refractive index at appctx->slice_rid
     Create a vector to hold base diag  at appctx->base_diag
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
  ierr = DMCreateGlobalVector(appctx.da,&appctx.slice_rid); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) appctx.slice_rid,"ref_index");CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(appctx.da,&appctx.base_diag); CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Store solution as hdf5
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"solution.h5",
                             FILE_MODE_WRITE,&appctx.hdf5_sol_viewer);CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures for solution
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
  ierr = DMCreateGlobalVector(appctx.da,&u); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u, "sol_vec");CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set matrix evaluation routine.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = DMCreateMatrix(appctx.da,&A); CHKERRQ(ierr);
  ierr = formMatrix(appctx.da,A,&appctx);

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set optional user-defined monitoring routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  //ierr = TSMonitorSet(ts,Monitor,&appctx,NULL);CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set time dependent linear RHS function.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,RHSMatrixMatter,&appctx);CHKERRQ(ierr);

 /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solution vector and initial timestep
 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSSetTimeStep(ts,appctx.step_time);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSSetMaxSteps(ts,appctx.prop_steps);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,prop_distance);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSRHSJacobianSetReuse(ts,ts_jac_reuse); CHKERRQ(ierr);

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

  ierr = TSView(ts,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = VecView(u,appctx.hdf5_sol_viewer);CHKERRQ(ierr);

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.slice_rid);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&appctx.hdf5_sol_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&appctx.hdf5_rid_viewer);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  ierr = PetscFinalize();
  return ierr;
}

