
static char help[] ="X-ray propagation in matter in 2D\n\
Assuming that the refractive indices are same at each slice\n\
Solves a simple time-independent linear PDE .\n\
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

/*
   Include "petscts.h" so that we can use TS solvers.  Note that this file
   automatically includes:
     petscsys.h     - base PETSc routines  
     petscvec.h     - vectors
     petscmat.h     - matrices
     petscis.h      - index sets            
     petscksp.h     - Krylov subspace methods
     petscpc.h      - preconditioners
     petscviewer.h  - viewers               
     petscksp.h     - linear solvers
     petscsnes.h    - nonlinear solvers
*/

#include <petscts.h>
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
  PetscReal   energy;            /* energy in ev */  
  PetscReal   step_time;         /* step size in time */
  PetscViewer hdf5_sol_viewer;   /* viewer to write the solution to hdf5*/
  Vec         slice_rid;         /* vector to hold the refractive index */
  Vec         base_diag;         /* vector to hold the diag at t=0 */

} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode InitialMatrix(Mat,AppCtx*);
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode RHSMatrixMatter(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);

int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  TS             ts;                     /* timestepping context */
  KSP            ksp;                    /* Krylov solver context */  
  Mat            A;                      /* matrix data structure */
  Vec            u;                      /* approximate solution vector */
  PetscReal      prop_distance;          /* propagation distance */
  PetscInt       prop_steps;             /* number of steps for propagation */
  PetscInt       steps;                  /* output for TSGetStepNumber */  
  PetscInt       M;                      /* total grid size : mx * my */
  PetscMPIInt    size,rank;              /* MPI size and rank*/
  PetscBool      ts_jac_reuse = true;    /* for TSRHSJacobianSetReuse*/
  PetscErrorCode ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  appctx.mx        = 5000;
  appctx.my        = 5000;  
  ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&appctx.mx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-my",&appctx.my,NULL);CHKERRQ(ierr);  
  M = appctx.mx*appctx.my; 

  appctx.energy    = 20000;  
  ierr = PetscOptionsGetReal(NULL,NULL,"-energy",&appctx.energy,NULL);
    CHKERRQ(ierr);    
  appctx.lambda    = (1239.84/appctx.energy)*1e-9;

  prop_distance   = 2e-6;
  ierr = PetscOptionsGetReal(NULL,NULL,"-prop_distance",&prop_distance,NULL);CHKERRQ(ierr);    
  
  prop_steps      = 4;
  ierr = PetscOptionsGetInt(NULL,NULL,"-prop_steps",&prop_steps,NULL);CHKERRQ(ierr);      
    
  appctx.step_grid_x = 1.5e-8;   
  appctx.step_grid_y = 1.5e-8;     
  ierr = PetscOptionsGetReal(NULL,NULL,"-step_grid_x",
                             &appctx.step_grid_x,NULL);CHKERRQ(ierr);    
  ierr = PetscOptionsGetReal(NULL,NULL,"-step_grid_y",
                             &appctx.step_grid_y,NULL);CHKERRQ(ierr);        
    
  appctx.step_time   = prop_distance/prop_steps;

  if(rank==0){
      ierr = PetscPrintf(PETSC_COMM_SELF,
                         "Solving a linear TS problem on %d processors\n",size);
      CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"mx : %d, my: %d, energy(in eV) : %e\n",
                         appctx.mx, appctx.my, appctx.energy);CHKERRQ(ierr);
      }
    
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create a vector to hold refractive index at appctx->slice_rid
     Create a vector to hold diag at t=0 at appctx->base_diag
     Destroy the viewer
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
  PetscViewer hdf_ref_index_viewer;
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"ref_index.h5",
                             FILE_MODE_READ,&hdf_ref_index_viewer);
    CHKERRQ(ierr);  

  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,
                      appctx.mx*appctx.my,&appctx.slice_rid);CHKERRQ(ierr);
    
  ierr = PetscObjectSetName((PetscObject) appctx.slice_rid,"ref_index");CHKERRQ(ierr); 
  
  ierr = VecLoad(appctx.slice_rid,hdf_ref_index_viewer);CHKERRQ(ierr);
  
  ierr = PetscViewerDestroy(&hdf_ref_index_viewer);CHKERRQ(ierr);  

    
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,
                      appctx.mx*appctx.my,&appctx.base_diag);CHKERRQ(ierr);  
    
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Store solution as hdf5
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"solution.h5",
                             FILE_MODE_WRITE,&appctx.hdf5_sol_viewer);CHKERRQ(ierr);
    
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures for solution
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,M,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u, "sol_vec");CHKERRQ(ierr);
 
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
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,M);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = InitialMatrix(A,&appctx);  
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

  ierr = TSSetMaxSteps(ts,prop_steps);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,prop_distance);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSRHSJacobianSetReuse(ts,ts_jac_reuse); CHKERRQ(ierr);  
    
  ierr = TSGetKSP(ts, &ksp); CHKERRQ(ierr);
    
  PetscReal abstol = 1e-12;  
  ierr = KSPSetTolerances(ksp,PETSC_DEFAULT,abstol,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
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
  ierr = VecDestroy(&appctx.slice_rid);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.base_diag);CHKERRQ(ierr);
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
   InitialMatrix - Computes the matrix at the initial time, saves the
                   diagonal of the matrix.

   Input Parameter:
   A - Jacobian amtrix
   appctx - user-defined application context
*/

PetscErrorCode InitialMatrix(Mat A,AppCtx* appctx)
{
    
  PetscErrorCode ierr;                    /* error code */  
  PetscInt       i,start,end;             /* i-> iteration number, start/end -> local row numbers */  
  PetscScalar    v;                       /* used to set matrix entries */
  PetscInt       row,col;                    
  PetscInt       set_row,set_col;
  PetscInt       Mx = appctx->mx;
  PetscInt       My = appctx->my;
  PetscReal      hx = appctx->step_grid_x;
  PetscReal      hy = appctx->step_grid_y;  
  PetscComplex   prefac = (-1*PETSC_i*appctx->lambda/(4*PETSC_PI));
 
   ierr = MatGetOwnershipRange(A,&start,&end); CHKERRQ(ierr);  
    
  for (i=start; i<end; i++) {
      row = i/Mx;
      col = i - row*Mx;
      if (row>0 && row<Mx-1 && col!=0 && col!=My-1){
          v = prefac*1/(hy*hy);
          set_row = i; set_col = i - My;
          ierr = MatSetValues(A,1,&set_row,1,&set_col,&v,INSERT_VALUES);
          CHKERRQ(ierr);
          }
      if (row>0 && row<Mx-1 && col!=0 && col!=My-1){
          v = prefac*1/(hy*hy);
          set_row = i; set_col = i + My;
          ierr = MatSetValues(A,1,&set_row,1,&set_col,&v,INSERT_VALUES);
          CHKERRQ(ierr);
          }
      if (col>0 && col<My-1 && row!=0 && row!=Mx-1){
          v = prefac*1/(hx*hx);
          set_row = i; set_col = i - 1;
          ierr = MatSetValues(A,1,&set_row,1,&set_col,&v,INSERT_VALUES);
          CHKERRQ(ierr);
          }
      if (col>0 && col<My-1 && row!=0 && row!=Mx-1){
          v = prefac*1/(hx*hx);
          set_row = i; set_col = i + 1;
          ierr = MatSetValues(A,1,&set_row,1,&set_col,&v,INSERT_VALUES);
          CHKERRQ(ierr);
          }
      
      set_row = i; set_col = i;
      if( row==0 || row==appctx->mx -1 || col==0 || col==appctx->my -1){
          v = 1;
          }
      else{v = -prefac*2/(hx*hx)-prefac*2/(hy*hy);}

      ierr = MatSetValues(A,1,&set_row,1,&set_col,&v,INSERT_VALUES);
      CHKERRQ(ierr);
      }
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Complete the matrix assembly process and set some options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  /*Set and option to indicate that we will never add a new nonzero location
    to the matrix. If we do, it will generate an error.*/
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);   
    
  ierr = MatGetDiagonal(A,appctx->base_diag); CHKERRQ(ierr);
    
  return 0;  
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
  PetscScalar    val;
  PetscErrorCode ierr;
  PetscInt       i,row,col;
  PetscInt       low,high;
    
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Get local vector storage info, set values and assemble
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  ierr = VecGetOwnershipRange(u,&low,&high); CHKERRQ(ierr);
  
  /*Plane Wave*/  
  for (i=low; i<high; i++) {
      val = 1;
      ierr = VecSetValues(u,1,&i,&val,INSERT_VALUES);
      }

  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);  

  return 0;
}

/* --------------------------------------------------------------------- */
/*
   Monitor - save solution computed at each timestep to hdf5 file.

   Input Parameters:
   ts     - the timestep context
   step   - the count of the current step 
   time   - the current time
   u      - the solution at this timestep
   ctx    - user-provided context
            
*/
PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal time,Vec u,void *ctx)
{
  AppCtx         *appctx = (AppCtx*) ctx;   /* user-defined application context */
  PetscErrorCode ierr;
  PetscInt       iteration_number;
  iteration_number = time/appctx->step_time;     

  ierr = PetscViewerHDF5SetTimestep(appctx->hdf5_sol_viewer,
                                    iteration_number);CHKERRQ(ierr);
  ierr = VecView(u,appctx->hdf5_sol_viewer);CHKERRQ(ierr);
    
  return 0;
}
/* --------------------------------------------------------------------- */
/*
   RHSMatrixMatter - User-provided routine to compute the time dependent
   right-hand-side matrix for FD wave propagation. 
   ->set matrix structure
   ->get refractive index for current slice
   ->add refractive index to matrix diagonal

   Input Parameters:
   ts  - the TS context
   t   - current time
   ctx - user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
   AA - Jacobian matrix
   BB - optionally different preconditioning matrix
   str - flag indicating matrix structure
*/
PetscErrorCode RHSMatrixMatter(TS ts,PetscReal t,Vec X,Mat AA,Mat BB,void *ctx)
{
  Mat            A       = AA;            /* Jacobian matrix */
  AppCtx         *appctx = (AppCtx*)ctx;  /* user-defined application context */
  PetscErrorCode ierr;                    /* error code */  
  PetscInt iteration_number;              /* get current iteration number */
  iteration_number = t/appctx->step_time;    
  

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Insert the diagonal from initial time,
     Add to it the current time dependent F
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDiagonalSet(A,appctx->base_diag,INSERT_VALUES);CHKERRQ(ierr);  
  ierr = MatDiagonalSet(A,appctx->slice_rid,ADD_VALUES);CHKERRQ(ierr);
    
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Complete the matrix assembly process and set some options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    
  return 0;
}
