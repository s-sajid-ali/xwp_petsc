
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
  PetscInt    prop_steps;        /* number of steps for propagation */
  PetscViewer hdf5_sol_viewer;   /* viewer to write the solution to hdf5*/
  Vec         slice_rid;         /* vector to hold the refractive index */
  Vec         base_diag;         /* vector to hold the diag at t=0 */  
  DM          da;                /* Use DMDA to manage grid and vecs*/
    
} AppCtx;

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
  KSP            ksp;                    /* Krylov solver context */  
  Mat            A;                      /* matrix data structure */
  Vec            u;                      /* approximate solution vector */
  PetscReal      prop_distance;          /* propagation distance */

  PetscInt       steps;                  /* output for TSGetStepNumber */  
  PetscInt       mx,my;                  /* grid size in x and y */  
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

  mx        = 4096;
  my        = 4096;  
  ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&mx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-my",&my,NULL);CHKERRQ(ierr);  
  M = mx*my; 

  appctx.energy    = 25000;  
  ierr = PetscOptionsGetReal(NULL,NULL,"-energy",&appctx.energy,NULL);
    CHKERRQ(ierr);    
  appctx.lambda    = (1239.84/appctx.energy)*1e-9;

  prop_distance   = 20e-6;
  ierr = PetscOptionsGetReal(NULL,NULL,"-prop_distance",&prop_distance,NULL);CHKERRQ(ierr);    
  
  appctx.prop_steps      = 10;
  ierr = PetscOptionsGetInt(NULL,NULL,"-prop_steps",&appctx.prop_steps,NULL);CHKERRQ(ierr);      
    
  appctx.L_x = 5e-10 * mx;   
  appctx.L_y = 5e-10 * my;     
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
                      DMDA_STENCIL_BOX,mx,my,
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
    MatStencil     row, col[9];             /* Stencil to set data */  
    PetscScalar    v[9];                    /* Space for storing values to be inserted to matrix*/
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
                //assume hx==hy
                v[0] = -prefac*20/(6*hx*hy); // interior: build a row
                if (i-1 > 0) {
                    col[ncols].j = j;    col[ncols].i = i-1;
                    v[ncols++] = prefac*4/(6*hx*hx);}
                if (i+1 < info.mx-1) {
                    col[ncols].j = j;    col[ncols].i = i+1;
                    v[ncols++] = prefac*4/(6*hx*hx);}
                if (j-1 > 0) {
                    col[ncols].j = j-1;  col[ncols].i = i;
                    v[ncols++] = prefac*4/(6*hy*hy);}
                if (j+1 < info.my-1) {
                    col[ncols].j = j+1;  col[ncols].i = i;
                    v[ncols++] = prefac*4/(6*hy*hy);}
                if (i-1 > 0 && j-1 > 0 ) {
                    col[ncols].j = j-1;    col[ncols].i = i-1;
                    v[ncols++] = prefac*1/(6*hx*hy);}
                if (i+1 < info.mx-1 > 0 && j+1 < info.my-1 ) {
                    col[ncols].j = j+1;    col[ncols].i = i+1;
                    v[ncols++] = prefac*1/(6*hx*hy);}
                if (i-1 > 0 && j+1 < info.my-1 ) {
                    col[ncols].j = j+1;    col[ncols].i = i-1;
                    v[ncols++] = prefac*1/(6*hx*hy);}
                if (info.mx-1 > 0 && j-1 > 0 ) {
                    col[ncols].j = j-1;    col[ncols].i = i+1;
                    v[ncols++] = prefac*1/(6*hx*hy);}
                
            }
            ierr = MatSetValuesStencil(A,1,&row,ncols,col,v,INSERT_VALUES); CHKERRQ(ierr);
        }
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    
    
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
  PetscErrorCode ierr;

  VecSet(u,1.0);
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
*/
PetscErrorCode RHSMatrixMatter(TS ts,PetscReal t,Vec X,Mat AA,Mat BB,void *ctx)
{
  Mat            A       = AA;            /* Jacobian matrix */
  AppCtx         *appctx = (AppCtx*)ctx;  /* user-defined application context */
  PetscErrorCode ierr;                    /* error code */  
  PetscInt       iteration_number;        /* get current iteration number */
  char           fname[sizeof "ref_index_dmda.h5"];
  
  iteration_number = t/appctx->step_time;
  

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Insert the diagonal from initial time,
     Add to it the current time dependant F
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDiagonalSet(A,appctx->base_diag,INSERT_VALUES);CHKERRQ(ierr);  
  
  sprintf(fname,"ref_index_dmda.h5",(int) appctx->prop_steps,(int) iteration_number);
  
  if (iteration_number < appctx->prop_steps){
      PetscViewer hdf5_rid_viewer;   /* viewer to read refractive index*/

      ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fname,
                                 FILE_MODE_READ,&hdf5_rid_viewer);
          CHKERRQ(ierr); 
      ierr = VecLoad(appctx->slice_rid,hdf5_rid_viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&hdf5_rid_viewer);CHKERRQ(ierr);  

      ierr = MatDiagonalSet(A,appctx->slice_rid,ADD_VALUES);CHKERRQ(ierr);
      }
    
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Complete the matrix assembly process and set some options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    
  return 0;
}
