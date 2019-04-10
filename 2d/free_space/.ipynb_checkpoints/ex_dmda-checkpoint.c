
static char help[] ="X-ray propagation in matter in 2D\n\
Solves a simple time-independent linear PDE .\n\
Input parameters include:\n\
  -mx,my         : number of grid points\n\
  -energy        : energy of x-rays in ev\n\
  -prop_distance : distance of propagation\n\
  -prop_steps    : number of propagation steps\n\
  -step_grid_x/y : pixel size in x and y\n\n";

/* ------------------------------------------------------------------------
   This program solves the two-dimensional helmholtz equation:
   
       u_t = A*u_xx + A*u_yy;     
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
#include <petscdmda.h>
#include <petscviewerhdf5.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/
typedef struct {
  PetscReal   L_x;               /* Length along x */
  PetscReal   L_y;               /* Length along y */  
  PetscReal   lambda;            /* wavelength */
  PetscReal   energy;            /* energy in ev */  
  PetscReal   step_time;         /* step size in time */
  PetscViewer hdf5_sol_viewer;   /* viewer to write the solution to hdf5*/
  DM          da;                /* Use DMDA to manage grid and vecs*/
  
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode formMatrix(DM, Mat, void*);
extern PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);

int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  TS             ts;                     /* timestepping context */
  Mat            A;                      /* matrix data structure */
  Vec            u;                      /* approximate solution vector */
  PetscReal      prop_distance;          /* propagation distance */
  PetscInt       prop_steps;             /* number of steps for propagation */
  PetscInt       steps;                  /* output for TSGetStepNumber */
  PetscInt       mx,my;                  /* grid size in x and y */  
  PetscInt       M;                      /* total grid size : mx * my */
  PetscMPIInt    size,rank;              /* MPI size and rank*/
  PetscErrorCode ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  mx        = 1024;
  my        = 1024;  
  ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&mx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-my",&my,NULL);CHKERRQ(ierr);  
  M = mx*my; 

  appctx.energy    = 25000;  
  ierr = PetscOptionsGetReal(NULL,NULL,"-energy",&appctx.energy,NULL);
    CHKERRQ(ierr);    
  appctx.lambda      = (1239.84/appctx.energy)*1e-9;

  prop_distance   = 1e-5;
  ierr = PetscOptionsGetReal(NULL,NULL,"-prop_distance",&prop_distance,NULL);CHKERRQ(ierr);    
  
  prop_steps      = 25;
  ierr = PetscOptionsGetInt(NULL,NULL,"-prop_steps",&prop_steps,NULL);CHKERRQ(ierr);      
    
  appctx.L_x = 2e-9*1024;   
  appctx.L_y = 2e-9*1024;     
  ierr = PetscOptionsGetReal(NULL,NULL,"-L_x",
                             &appctx.L_x,NULL);CHKERRQ(ierr);    
  ierr = PetscOptionsGetReal(NULL,NULL,"-L_y",
                             &appctx.L_y,NULL);CHKERRQ(ierr);        
    
  appctx.step_time   = prop_distance/prop_steps;

  if(rank==0){
      ierr = PetscPrintf(PETSC_COMM_SELF,
                         "Solving a linear TS problem on %d processors\n",size);
      CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"mx : %d, my: %d, lambda : %e\n",
                         mx, my, appctx.lambda);CHKERRQ(ierr);
      }
    
    
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create 2D DMDA to manage grid and vecs.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                       DMDA_STENCIL_STAR,mx,my,
                       PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&appctx.da); CHKERRQ(ierr);

  ierr = DMSetFromOptions(appctx.da); CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da); CHKERRQ(ierr);
    
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
  ierr = DMCreateMatrix(appctx.da,&A); CHKERRQ(ierr);    
  ierr =  formMatrix(appctx.da,A,&appctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set time dependent linear RHS function. 
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,&appctx);CHKERRQ(ierr);
  
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
   
  PetscBool ts_jac_reuse=true;  
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
     Free work space.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&appctx.hdf5_sol_viewer);CHKERRQ(ierr);   
  
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
    PetscInt       i, j;
    PetscScalar    **ab;
    DMDALocalInfo  info;

    ierr = DMDAGetLocalInfo(appctx->da,&info); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(appctx->da, u, &ab);CHKERRQ(ierr);
    
    for (j=info.ys; j<info.ys+info.ym; j++) {
        for (i=info.xs; i<info.xs+info.xm; i++) {
            if (i > info.mx*3/8 && i < info.mx*5/8){
                if(j > info.my*3/8 && j < info.my*5/8){
                ab[j][i] = 1.0;  
                }
            }
            else {
                ab[j][i] = 0.0;
            }
        }
    }
    ierr = DMDAVecRestoreArray(appctx->da, u, &ab); CHKERRQ(ierr);
    
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
   formMatrix - Computes the matrix at the initial time, saves the
                   diagonal of the matrix.

   Input Parameter:
   da - DMDA object
   A  - Jacobian amtrix
   appctx - user-defined application context
*/

PetscErrorCode formMatrix(DM da, Mat A, void* ctx)
{
    AppCtx         *appctx = (AppCtx*)ctx;  /* user-defined application context */
    PetscErrorCode ierr;                    /* Error code */  
    DMDALocalInfo  info;                    /* For storing DMDA info */   
    MatStencil     row, col[5];             /* Stencil to set data */  
    PetscScalar    v[5];                    /* Space for storing values to be inserted to matrix*/
    PetscInt       i, j, ncols;             /* Iteration over local rows,cols and stencil cols */ 
    
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    
    PetscReal      hx = appctx->L_x/info.mx;
    PetscReal      hy = appctx->L_y/info.my;  
    PetscComplex   prefac = (-1*PETSC_i*appctx->lambda/(4*PETSC_PI));
        
    for (j = info.ys; j < info.ys+info.ym; j++) {
        for (i = info.xs; i < info.xs+info.xm; i++) {
            row.j = j;           // row of A corresponding to (x_i,y_j)
            row.i = i;
            col[0].j = j;        // diagonal entry
            col[0].i = i;
            ncols = 1;
            if (i==0 || i==info.mx-1 || j==0 || j==info.my-1) {
                v[0] = 1.0;      // on boundary: trivial equation
            } else {
                v[0] = -prefac*2/(hx*hx)-prefac*2/(hy*hy); // interior: build a row
                if (i-1 > 0) {
                    col[ncols].j = j;    col[ncols].i = i-1;
                    v[ncols++] = prefac*1/(hx*hx);}
                if (i+1 < info.mx-1) {
                    col[ncols].j = j;    col[ncols].i = i+1;
                    v[ncols++] = prefac*1/(hx*hx);}
                if (j-1 > 0) {
                    col[ncols].j = j-1;  col[ncols].i = i;
                    v[ncols++] = prefac*1/(hy*hy);}
                if (j+1 < info.my-1) {
                    col[ncols].j = j+1;  col[ncols].i = i;
                    v[ncols++] = prefac*1/(hy*hy);}
            }
            ierr = MatSetValuesStencil(A,1,&row,ncols,col,v,INSERT_VALUES); CHKERRQ(ierr);
        }
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    return 0;
}
