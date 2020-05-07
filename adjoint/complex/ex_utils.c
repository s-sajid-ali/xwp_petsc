
static char help[] ="Utilities \n";

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
  char           fname[sizeof "dmda_slice_100_100.h5"];
  
  iteration_number = t/appctx->step_time;
  

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Insert the diagonal from initial time,
     Add to it the current time dependant F
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDiagonalSet(A,appctx->base_diag,INSERT_VALUES);CHKERRQ(ierr);  
  
  sprintf(fname,"dmda_slice_%d_%d.h5",(int) appctx->prop_steps,(int) iteration_number);
  
  if (iteration_number < appctx->prop_steps){
      ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, fname,
                                 FILE_MODE_READ,&appctx->hdf5_rid_viewer);
          CHKERRQ(ierr); 
      ierr = VecLoad(appctx->slice_rid,appctx->hdf5_rid_viewer);CHKERRQ(ierr);
      ierr = MatDiagonalSet(A,appctx->slice_rid,ADD_VALUES);CHKERRQ(ierr);
      }
    
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Complete the matrix assembly process and set some options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    
  return 0;
}
