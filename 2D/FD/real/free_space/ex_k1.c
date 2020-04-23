static char help[] ="X-ray propagation in matter in 2D\n\
Assuming that the refractive indices are same at each slice\n\
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
   
       complex :
       u_t = A*u_xx + A*u_yy + F_t*u;     
       
       real : 
       a_t = A*a_xx + A*a_yy + f_r*a - f_i*b
       b_t = A*b_xx + A*b_yy + f_i*a + f_r*b
       
       This program executes the K1 approach from 
       https://doi.org/10.1137/S1064827500372262 : 
       
       ( 0    -\del   ( real_wave   = ( a_t
         \del  0   )    imag_wave )     b_t )
       
  ------------------------------------------------------------------------- */

#include <petscdmda.h>
#include <petscviewerhdf5.h>
#include <petscts.h>

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
  Vec         slice_rid;          /* vector to hold F */
  DM          da;                /* Use DMDA to manage grid and vecs*/
    
} AppCtx;

/*  Simple C struct that allows us to access the two velocity (x and y directions) values easily in the code */
typedef struct {
  PetscScalar a,b;
} Field;


/*
   User-defined routines
*/
extern PetscErrorCode InitialConditions(DM, Vec,AppCtx*);
extern PetscErrorCode formMatrix(DM, Mat, void*);
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

  mx        = 1024;
  my        = 1024;  
  ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&mx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-my",&my,NULL);CHKERRQ(ierr);  

  appctx.energy    = 25000;  
  ierr = PetscOptionsGetReal(NULL,NULL,"-energy",&appctx.energy,NULL);
    CHKERRQ(ierr);    
  appctx.lambda    = (1239.84/appctx.energy)*1e-9;

  prop_distance   = 1e-5;
  ierr = PetscOptionsGetReal(NULL,NULL,"-prop_distance",&prop_distance,NULL);CHKERRQ(ierr);    
  
  prop_steps      = 25;
  ierr = PetscOptionsGetInt(NULL,NULL,"-prop_steps",&prop_steps,NULL);CHKERRQ(ierr);      
    
  appctx.L_x = 2e-9 * mx;   
  appctx.L_y = 2e-9 * my;     
  ierr = PetscOptionsGetReal(NULL,NULL,"-L_x",
                             &appctx.L_x,NULL);CHKERRQ(ierr);    
  ierr = PetscOptionsGetReal(NULL,NULL,"-L_y",
                             &appctx.L_y,NULL);CHKERRQ(ierr);        
    
  appctx.step_time   = prop_distance/prop_steps;

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
    
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,
                      DMDA_STENCIL_STAR,mx,my,
                      PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,
                      &appctx.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(appctx.da); CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da); CHKERRQ(ierr);
    
  ierr = DMDASetFieldName(appctx.da,0,"a");CHKERRQ(ierr);
  ierr = DMDASetFieldName(appctx.da,1,"b");CHKERRQ(ierr);
    
  
 /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create a vector to hold refractive index at appctx->slice_rid
     Destroy the viewer
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
    
  ierr = DMCreateGlobalVector(appctx.da,&appctx.slice_rid); CHKERRQ(ierr);  
  ierr = PetscObjectSetName((PetscObject) appctx.slice_rid,"ref_index");CHKERRQ(ierr); 
  
  /*  
  PetscViewer hdf_slice_rid_viewer;
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"ref_index_dmda.h5",
                             FILE_MODE_READ,&hdf_slice_rid_viewer);CHKERRQ(ierr); 
  ierr = VecLoad(appctx.slice_rid,hdf_slice_rid_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&hdf_slice_rid_viewer);CHKERRQ(ierr);  
  */   
     
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures for solution
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
  ierr = DMCreateGlobalVector(appctx.da,&u); CHKERRQ(ierr); 
  ierr = PetscObjectSetName((PetscObject)u, "sol_vec");CHKERRQ(ierr);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set matrix evaluation routine.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 
  ierr = DMCreateMatrix(appctx.da,&A); CHKERRQ(ierr);
  ierr = MatSetBlockSize(A, 2); CHKERRQ(ierr);
  ierr = formMatrix(appctx.da,A,&appctx);  
  
  /*
     Evaluate initial conditions
  */
  ierr = InitialConditions(appctx.da, u, &appctx);CHKERRQ(ierr);    

  /*
  PetscViewer hdf5_init_viewer;
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"initial.h5",
                             FILE_MODE_WRITE,&hdf5_init_viewer);CHKERRQ(ierr);
  ierr = VecView(u,hdf5_init_viewer);CHKERRQ(ierr);  
  ierr = PetscViewerDestroy(&hdf5_init_viewer);CHKERRQ(ierr);
  */

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"solution.h5",
                             FILE_MODE_WRITE,&appctx.hdf5_sol_viewer);CHKERRQ(ierr);  
  

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);
  ierr = TSSetDM(ts,appctx.da);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,&appctx);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,appctx.step_time);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,prop_steps);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,prop_distance);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);  
  ierr = TSMonitorSet(ts,Monitor,&appctx,NULL);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSRHSJacobianSetReuse(ts,ts_jac_reuse); CHKERRQ(ierr);  

  ierr = TSSolve(ts,u);CHKERRQ(ierr);
    
 /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Store solution as hdf5
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  //ierr = VecView(u,appctx.hdf5_sol_viewer);CHKERRQ(ierr);  
    
    
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
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
    MatStencil     row, col[5];             /* Stencil to set data */  
    PetscScalar    v[5];                    /* Space for storing values to be inserted to matrix*/
    PetscInt       i, j, ncols;             /* Iteration over local rows,cols and stencil cols */ 
    PetscInt       dof;                     /* Flag to indicate field  */
    Field          **ref_idx;
    
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(da,appctx->slice_rid,&ref_idx);CHKERRQ(ierr);
    
    PetscReal      hx = appctx->L_x/info.mx;
    PetscReal      hy = appctx->L_y/info.my;  
    PetscReal      prefac = (appctx->lambda/(4*PETSC_PI));
    
    for (dof = 0; dof < 2; dof++) {
        if (dof==1){
            prefac = -1*(appctx->lambda/(4*PETSC_PI));
        }
        
        row.c = dof;
        col[0].c = 1-dof;
        col[1].c = 1-dof;
        col[2].c = 1-dof;
        col[3].c = 1-dof;
        col[4].c = 1-dof;
                
        for (j = info.ys; j < info.ys+info.ym; j++) {
            for (i = info.xs; i < info.xs+info.xm; i++) {
                
                row.j = j;           // row of A corresponding to (x_i,y_j)
                row.i = i;   
                col[0].j = j;        // diagonal entry
                col[0].i = i;
                ncols = 1;

                v[0] = -prefac*2/(hx*hx)-prefac*2/(hy*hy); // interior: build a row/column
                col[ncols].j = j;    col[ncols].i = i-1;
                v[ncols++] = prefac*1/(hx*hx);
                col[ncols].j = j;    col[ncols].i = i+1;
                v[ncols++] = prefac*1/(hx*hx);
                col[ncols].j = j-1;  col[ncols].i = i;
                v[ncols++] = prefac*1/(hy*hy);
                col[ncols].j = j+1;  col[ncols].i = i;
                v[ncols++] = prefac*1/(hy*hy);
                ierr = MatSetValuesStencil(A,1,&row,ncols,col,v,INSERT_VALUES); CHKERRQ(ierr);
                }
                
            } 
        }

    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Assembling matrix \n");
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Finished assembling matrix \n");
    
    ierr = DMDAVecRestoreArrayRead(da,appctx->slice_rid,&ref_idx);CHKERRQ(ierr);
    
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
PetscErrorCode InitialConditions(DM da, Vec u,AppCtx *appctx)
{
  PetscErrorCode ierr;
  PetscInt i,j;
  DMDALocalInfo  info;                    /* For storing DMDA info */   
  Field          **vals;
  
   
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,u,&vals);CHKERRQ(ierr);
    
  for (j = info.ys; j < info.ys+info.ym; j++) {
      for (i = info.xs; i < info.xs+info.xm; i++) {
          if (i>info.mx*3/8 && i< info.mx*5/8){
              if (j>info.mx*3/8 && j< info.mx*5/8){
                  vals[j][i].a = 1;
                  vals[j][i].b = 0;
                  }
              }
          }
      }

  ierr = DMDAVecRestoreArray(da,u,&vals);CHKERRQ(ierr);

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
