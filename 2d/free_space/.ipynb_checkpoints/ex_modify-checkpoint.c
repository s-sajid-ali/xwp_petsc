
static char help[] ="Attempt X-ray propagation in free space based on ex3\n\
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

   This program solves the two-dimensional helmholtz equation:
       u_t = A*u_xx + A*u_yy;
   This is a linear, second-order, parabolic equation.

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
  PetscInt    mx;                /* total number of grid points in x */
  PetscInt    my;                /* total number of grid points in y */  
  PetscReal   step_grid_x;       /* grid spacing in x */
  PetscReal   step_grid_y;       /* grid spacing in y */  
  PetscReal   lambda;            /* wavelength */
  PetscReal   step_time;         /* step size in time */
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
  PetscReal      time_total_max = 1e-4;  /* default max total time */
  PetscInt       time_steps_max = 512;  /* default max timesteps */
  PetscInt       steps;                  /* output for TSGetStepNumber */  
  PetscInt       mx;                     /* problem size in x*/
  PetscInt       my;                     /* problem size in y*/
  PetscInt       M;                      /* mx * my */
  PetscReal      dt;                     /* For TSSetTimeStep*/
  PetscMPIInt    size;                   /* MPI size*/
  PetscErrorCode ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  mx  = 512;
  my  = 512;    
  ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&mx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-my",&my,NULL);CHKERRQ(ierr);  
    
  M = mx*my;  

  appctx.mx          = mx;
  appctx.my          = my;  
  appctx.step_grid_x = 1.4e-9; 
  appctx.step_grid_y = 1.4e-9;   
  appctx.lambda      = 1.23984e-10;
  appctx.step_time   = time_total_max/time_steps_max;

  ierr = PetscPrintf(PETSC_COMM_SELF,"Solving a linear TS problem on 1 processor\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"mx : %d, my: %d lambda : %e\n",appctx.mx, appctx.my, appctx.lambda);CHKERRQ(ierr);
    
    
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
  ierr = VecCreateSeq(PETSC_COMM_SELF,M,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u, "sol_vec");CHKERRQ(ierr);
 
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
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,M);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

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
  PetscScalar    *u_localptr;
  PetscErrorCode ierr;
  PetscInt       i,j,k;
    

  /*
    Get a pointer to vector data.
    - For default PETSc vectors, VecGetArray() returns a pointer to
      the data array.  Otherwise, the routine is implementation dependent.
    - You MUST call VecRestoreArray() when you no longer need access to
      the array.
  */
  ierr = VecGetArray(u,&u_localptr);CHKERRQ(ierr);

  /*
     We initialize the solution array by simply writing the solution
     directly into the array locations.  Alternatively, we could use
     VecSetValues() or VecSetValuesLocal().
  */
  for (i=0; i<appctx->mx; i++) {
      for (j=0; j<appctx->my; j++){
          k = i*appctx->mx+j;
          if (i<appctx->mx*3/8){u_localptr[k] = 0+0*PETSC_i;}
          else if (i>((appctx->mx*5/8))){u_localptr[k] = 0+0*PETSC_i;}
          else {
              if(j<appctx->my*3/8){u_localptr[k] = 0+0*PETSC_i;}
              else if (j>((appctx->my*5/8))){u_localptr[k] = 0+0*PETSC_i;}
              else {u_localptr[k] = 1+0*PETSC_i;}
                   //ierr = PetscPrintf(PETSC_COMM_SELF,"i:%d, j : %d, k : %d\n",i,j,k);}
              }
          }
      }
  /*
     Restore vector
  */
  ierr = VecRestoreArray(u,&u_localptr);CHKERRQ(ierr);


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
  PetscInt       iteration_number;
  iteration_number = time/appctx->step_time;    
    
    
  ierr = PetscViewerHDF5SetTimestep(appctx->hdf5_sol_viewer, iteration_number);CHKERRQ(ierr);
  ierr = VecView(u,appctx->hdf5_sol_viewer);CHKERRQ(ierr);
    
  return 0;
}
/* --------------------------------------------------------------------- */
/*
   RHSMatrixFreeSpace - User-provided routine to compute the right-hand-side
   matrix for the helmholtz equation.

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
  PetscInt       mstart  = appctx->my;
  PetscInt       mend    = (appctx->mx-1)*appctx->my;
  PetscErrorCode ierr;   
  PetscInt       i,idx[5];
  PetscInt       loc_row,loc_col;
  PetscScalar    v[5];  
  Vec            v_diag;
  PetscComplex   prefac = (-1*PETSC_i*appctx->lambda/(4*PETSC_PI));
  PetscComplex   hx = appctx->step_grid_x;
  PetscComplex   hy = appctx->step_grid_y;  
  
  PetscInt iteration_number;
  iteration_number = t/appctx->step_time;    
    
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute entries for the locally owned part of the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Set matrix rows corresponding to interior data.  We construct the
     matrix one row at a time.
  */
  for (i=mstart; i<mend; i++) {
       loc_col = i%appctx->mx;
       loc_row = i/appctx->mx;
       if (loc_col != 0){
           if(loc_col != appctx->my-1){
                 
                 idx[0] = (loc_row-1)*appctx->my + loc_col;
                 idx[1] = loc_row*appctx->my + loc_col - 1 ;
                 idx[2] = loc_row*appctx->my + loc_col ;
                 idx[3] = loc_row*appctx->my + loc_col + 1 ;
                 idx[4] = (loc_row+1)*appctx->my + loc_col ;
                 
                 v[0] = prefac*1/(hy*hy);
                 v[1] = prefac*1/(hx*hx);
                 v[2] = -prefac*2/(hx*hx)-prefac*2/(hy*hy);
                 v[3] = prefac*1/(hx*hx);
                 v[4] = prefac*1/(hy*hy);
                 
                 ierr = MatSetValues(A,1,&i,5,idx,v,INSERT_VALUES);CHKERRQ(ierr);
                
                 /*if (t==0){
                     PetscPrintf(PETSC_COMM_SELF,"i:%d, row : %d, col : %d\n",i,loc_row,loc_col);
                     PetscPrintf(PETSC_COMM_SELF,"%d %d %d %d %d\n\n",idx[0],idx[1],idx[2],idx[3],idx[4]);
                     }*/
               }
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
    
    
   
  ierr = VecCreateSeq(PETSC_COMM_SELF,appctx->mx*appctx->my,&v_diag);CHKERRQ(ierr);
  ierr = VecSet(v_diag,0+0*PETSC_i); CHKERRQ(ierr);
  ierr = MatDiagonalSet(A,v_diag,ADD_VALUES); CHKERRQ(ierr); 
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   
  
    
  /*
     Set and option to indicate that we will never add a new nonzero location
     to the matrix. If we do, it will generate an error.
  */
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);   
    
  return 0;
}
