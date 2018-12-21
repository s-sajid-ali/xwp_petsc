
static char help[] ="Attempt X-ray propagation in matter based on ex3\n\
Clear unnecesary norm checking\n\
Solves a simple time-dependent linear PDE .\n\
Input parameters include:\n\
  -m <points>, where <points> = number of grid points\n\
  -debug              : Activate debugging printouts\n\
  -nox                : Deactivate x-window graphics\n\n";

/*
   Concepts: TS^time-dependent linear problems
   Concepts: TS^diffusion equation
   Processors: 1
*/

/* ------------------------------------------------------------------------

   This program solves the one-dimensional helmholtz equation:
       u_t = A*u_xx + F*u,
   This is a linear, second-order, parabolic equation.

   We discretize the right-hand side using finite differences with
   uniform grid spacing h:
       u_xx = (u_{i+1} - 2u_{i} + u_{i-1})/(h^2)
   We then demonstrate time evolution using the various TS methods by
   running the program via
       ex3 -ts_type <timestepping solver>

   Notes:
   time-dependent F:   f(u,t) is a function of t

    Uniprocessor example

  ------------------------------------------------------------------------- */

/*
   Include "petscts.h" so that we can use TS solvers.  Note that this file
   automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h  - vectors
     petscmat.h  - matrices
     petscis.h     - index sets            petscksp.h  - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h   - preconditioners
     petscksp.h   - linear solvers        petscsnes.h - nonlinear solvers
*/

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
  PetscReal   lambda;            /* wavelength */
  PetscReal   step_time;         /* step size in time */
  PetscBool   debug;             /* flag (1 indicates activation of debugging printouts) */
  PetscViewer viewer1;           /* viewer for the solution */
  PetscViewer    hdf5_sol_viewer;        /* viewer to write the solution to hdf5*/
  Mat         ref_index;         /* Matrix holding the refractive indices*/
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode RHSMatrixMatter(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);

int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  TS             ts;                     /* timestepping context */
  Mat            A;                      /* matrix data structure */
  Vec            u;                      /* approximate solution vector */
  PetscReal      time_total_max = 1e-3;  /* default max total time */
  PetscInt       time_steps_max = 5000;  /* default max timesteps */
  PetscDraw      draw;                   /* drawing context */  
  PetscInt       steps;                  /* output for TSGetStepNumber */  
  PetscInt       m;                      /* problem size */
  PetscReal      dt;                     /* For TSSetTimeStep*/
  PetscMPIInt    size;                   /* MPI size*/
  PetscErrorCode ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  m    = 5000;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-debug",&appctx.debug);CHKERRQ(ierr);

  appctx.m         = m;
  appctx.step_grid = 1.4e-9;  
  appctx.lambda    = 1.23984e-10;
  appctx.step_time = time_total_max/time_steps_max;

  ierr = PetscPrintf(PETSC_COMM_SELF,"Solving a linear TS problem on 1 processor\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"m : %d, lambda : %e\n",appctx.m, appctx.lambda);CHKERRQ(ierr);
    
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Read the refractive index matrix and save it at appctx->ref_index
     Destroy the viewer
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscViewer ref_index_viewer;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"refractive_index_bin.dat",FILE_MODE_READ,&ref_index_viewer);CHKERRQ(ierr);
  
  ierr = MatCreate(PETSC_COMM_WORLD,&appctx.ref_index);CHKERRQ(ierr);
  ierr = MatSetType(appctx.ref_index,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(appctx.ref_index);CHKERRQ(ierr);
  ierr = MatLoad(appctx.ref_index,ref_index_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&ref_index_viewer);CHKERRQ(ierr);
    
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Store solution as hdf5
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"solution.h5",FILE_MODE_WRITE,&appctx.hdf5_sol_viewer);CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures for approximate solutions
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
  ierr = VecCreateSeq(PETSC_COMM_SELF,m,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u, "Sol_vec");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set up display to show graph of the solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"",80,380,400,160,&appctx.viewer1);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDraw(appctx.viewer1,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);
 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set optional user-defined monitoring routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSMonitorSet(ts,Monitor,&appctx,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set matrix evaluation routine.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,m);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  /*
    For linear problems with a time-dependent f(u,t) in the equation
    u_t = f(u,t), the user provides the discretized right-hand-side
    as a time-dependent matrix.
  */
  ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,RHSMatrixMatter,&appctx);CHKERRQ(ierr);
  
 /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solution vector and initial timestep
 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  dt   = time_total_max/time_steps_max;
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize timestepping solver:
       - Set the solution method to be the Backward Euler method.
       - Set timestepping duration info
     Then set runtime options, which can override these defaults.
     For example,
          -ts_max_steps <maxsteps> -ts_final_time <maxtime>
     to override the defaults set by TSSetMaxSteps()/TSSetMaxTime().
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSSetMaxSteps(ts,time_steps_max);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,time_total_max);CHKERRQ(ierr);
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
  ierr = PetscViewerDestroy(&appctx.viewer1);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&appctx.hdf5_sol_viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.ref_index);CHKERRQ(ierr);

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
  PetscScalar    *u_localptr;
  PetscErrorCode ierr;
  PetscInt       i;
    

  /*
    Get a pointer to vector data.
    - For default PETSc vectors, VecGetArray() returns a pointer to
      the data array.  Otherwise, the routine is implementation dependent.
    - You MUST call VecRestoreArray() when you no longer need access to
      the array.
    - Note that the Fortran interface to VecGetArray() differs from the
      C version.  See the users manual for details.
  */
  ierr = VecGetArray(u,&u_localptr);CHKERRQ(ierr);

  /*
     We initialize the solution array by simply writing the solution
     directly into the array locations.  Alternatively, we could use
     VecSetValues() or VecSetValuesLocal().
  */
  for (i=0; i<appctx->m; i++) {
      if (i<appctx->m/7){u_localptr[i] = 0+0*PETSC_i;}
      else if (i>((appctx->m*3/7))){u_localptr[i] = 0+0*PETSC_i;}
      else {u_localptr[i] = 1+0*PETSC_i;}
      }
  /*
     Restore vector
  */
  ierr = VecRestoreArray(u,&u_localptr);CHKERRQ(ierr);

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
  PetscReal      dt,dttol;
  Vec            u_abs;                  /* absolute value of approximate solution vector */
  PetscInt       iteration_number;
  iteration_number = time/appctx->step_time;    
    
  ierr = PetscViewerHDF5SetTimestep(appctx->hdf5_sol_viewer, iteration_number);CHKERRQ(ierr);
  ierr = VecView(u,appctx->hdf5_sol_viewer);CHKERRQ(ierr);
   
    
  /*- - - - - - - - - - - - - - - - - - - -
      Copy solution vector to new vector, 
      conver to absolute value for viewing
   - - - - - - - - - - - - - - - - - - - -*/
  ierr = VecDuplicate(u,&u_abs);CHKERRQ(ierr);
  ierr = VecCopy(u,u_abs);CHKERRQ(ierr);
  ierr = VecAbs(u_abs);CHKERRQ(ierr);
    
  /* - - - - - - - - - - - - - - - - - - - - 
      View a graph of the current iterate
   - - - - - - - - - - - - - - - - - - - - */
  ierr = VecView(u_abs,appctx->viewer1);CHKERRQ(ierr);
  

  ierr = VecDestroy(&u_abs);CHKERRQ(ierr);
      
  /*
     Print debugging information if desired
  */
  if (appctx->debug) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Computed solution vector\n");CHKERRQ(ierr);
    ierr = VecView(u,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
     }

  return 0;
}
/* --------------------------------------------------------------------- */
/*
   RHSMatrixHeat - User-provided routine to compute the right-hand-side
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
PetscErrorCode RHSMatrixMatter(TS ts,PetscReal t,Vec X,Mat AA,Mat BB,void *ctx)
{
  Mat            A       = AA;                /* Jacobian matrix */
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscInt       mstart  = 0;
  PetscInt       mend    = appctx->m;
  PetscErrorCode ierr;
  PetscInt       i,idx[3];
  PetscScalar    v[3];  
  Vec            slice_vec;
  PetscComplex   prefac = (-1*PETSC_i*appctx->lambda/(4*PETSC_PI))*(1/(appctx->step_grid*appctx->step_grid));

  
  PetscInt iteration_number;
  iteration_number = t/appctx->step_time;    
    
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute entries for the locally owned part of the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Set matrix rows corresponding to boundary data
  */

  mstart = 0;
  v[0]   = 1.0;
  ierr   = MatSetValues(A,1,&mstart,1,&mstart,v,INSERT_VALUES);CHKERRQ(ierr);
  mstart++;

  mend--;
  v[0] = 1.0;
  ierr = MatSetValues(A,1,&mend,1,&mend,v,INSERT_VALUES);CHKERRQ(ierr);

  /*
     Set matrix rows corresponding to interior data.  We construct the
     matrix one row at a time.
  */
  v[0] = prefac*-1; v[1] = prefac*2; v[2] = prefac*-1;
  for (i=mstart; i<mend; i++) {
    idx[0] = i-1; idx[1] = i; idx[2] = i+1;
    ierr   = MatSetValues(A,1,&i,3,idx,v,INSERT_VALUES);CHKERRQ(ierr);
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

    
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set the diagonal with current time dependant F
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
  ierr = VecCreateSeq(PETSC_COMM_SELF,appctx->m,&slice_vec);CHKERRQ(ierr);
  if (iteration_number<appctx->m){
      ierr = MatGetColumnVector(appctx->ref_index,slice_vec,iteration_number);CHKERRQ(ierr);}
  ierr = MatDiagonalSet(A,slice_vec,ADD_VALUES); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Boundary conditions.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  PetscInt     row[2];
  PetscScalar  rho;
  row[0]=0; row[1]=appctx->m-1; 
  rho = 0;
  ierr = MatZeroRowsColumns(A, 2, row, rho, NULL,NULL ); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
    
  /*
     Set and option to indicate that we will never add a new nonzero location
     to the matrix. If we do, it will generate an error.
  */
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  return 0;
}

/*TEST

    test:
      args: -nox -ts_type ssp -ts_dt 0.0005

    test:
      suffix: 2
      args: -nox -ts_type ssp -ts_dt 0.0005 -time_dependent_rhs 1

    test:
      suffix: 3
      args:  -nox -ts_type rosw -ts_max_steps 3 -ksp_converged_reason
      filter: sed "s/ATOL/RTOL/g"
      requires: !single

    test:
      suffix: 4
      args: -nox -ts_type beuler -ts_max_steps 3 -ksp_converged_reason
      filter: sed "s/ATOL/RTOL/g"

    test:
      suffix: 5
      args: -nox -ts_type beuler -ts_max_steps 3 -ksp_converged_reason -time_dependent_rhs
      filter: sed "s/ATOL/RTOL/g"

    test:
      requires: !single
      suffix: pod_guess
      args: -nox -ts_type beuler -use_ifunc -ts_dt 0.0005 -ksp_guess_type pod -pc_type none -ksp_converged_reason

    test:
      requires: !single
      suffix: pod_guess_Ainner
      args: -nox -ts_type beuler -use_ifunc -ts_dt 0.0005 -ksp_guess_type pod -ksp_guess_pod_Ainner -pc_type none -ksp_converged_reason

    test:
      requires: !single
      suffix: fischer_guess
      args: -nox -ts_type beuler -use_ifunc -ts_dt 0.0005 -ksp_guess_type fischer -pc_type none -ksp_converged_reason

    test:
      requires: !single
      suffix: fischer_guess_2
      args: -nox -ts_type beuler -use_ifunc -ts_dt 0.0005 -ksp_guess_type fischer -ksp_guess_fischer_model 2,10 -pc_type none -ksp_converged_reason
TEST*/
