static char help[] = "Rotate a 3D object stored as 2D matrix!\n\n";

/* Utilities for rotation of 3D object stored as 2D matrix! */

#include <petscmat.h>
#include <petscviewerhdf5.h>
#include "appctx.h"


/* ---------------------------------------------------------------------
   initialize - Initialize the problem.
   AppCtx - user-defined application context
 --------------------------------------------------------------------- */
PetscErrorCode initialize(void *ctx)
{
    PetscErrorCode ierr;
    AppCtx         *appctx = (AppCtx*) ctx;   /* user-defined application context */

    appctx->debug_flag  = PETSC_TRUE;
    appctx->cols        = 64;
    PetscInt _cols = appctx->cols;

    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&appctx->rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&appctx->size);CHKERRQ(ierr);

    /* Create (sparse) rotation matrix */
    ierr = MatCreate(PETSC_COMM_WORLD,&appctx->rot);CHKERRQ(ierr);
    ierr = MatSetType(appctx->rot,MATMPIAIJ);CHKERRQ(ierr);
    ierr = MatSetSizes(appctx->rot,PETSC_DECIDE,PETSC_DECIDE,_cols*_cols,_cols*_cols);CHKERRQ(ierr);
    ierr = MatSetFromOptions(appctx->rot);CHKERRQ(ierr);

    /* Create (dense) refractive index matrix */
    ierr = MatCreate(PETSC_COMM_WORLD,&appctx->beta_in);CHKERRQ(ierr);
    ierr = MatSetType(appctx->beta_in,MATDENSE);CHKERRQ(ierr);
    ierr = MatSetSizes(appctx->beta_in,PETSC_DECIDE,PETSC_DECIDE,_cols*_cols,_cols);CHKERRQ(ierr);
    ierr = MatSetFromOptions(appctx->beta_in);CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_WORLD,&appctx->beta_out);CHKERRQ(ierr);
    ierr = MatSetType(appctx->beta_out,MATDENSE);CHKERRQ(ierr);
    ierr = MatSetSizes(appctx->beta_out,PETSC_DECIDE,PETSC_DECIDE,_cols*_cols,_cols);CHKERRQ(ierr);
    ierr = MatSetFromOptions(appctx->beta_out);CHKERRQ(ierr);
    ierr = MatSetUp(appctx->beta_out); CHKERRQ(ierr);

    /* Create work vectors */
    ierr = VecCreateMPI(PETSC_COMM_WORLD,((_cols*_cols)/(appctx->size)),PETSC_DECIDE,&appctx->X); CHKERRQ(ierr);
    ierr = VecSetBlockSize(appctx->X, _cols); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)appctx->X, "X_vec");CHKERRQ(ierr);
    ierr = VecCreateMPI(PETSC_COMM_WORLD,((_cols*_cols)/(appctx->size)),PETSC_DECIDE,&appctx->Y); CHKERRQ(ierr);
    ierr = VecSetBlockSize(appctx->X, _cols); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)appctx->Y, "Y_vec");CHKERRQ(ierr);

    /* Load refractive indices */
    /*remove beta_out_view after debugging!*/
    PetscViewer    beta_in_view;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,\
                                 "beta.dat",\
                                 FILE_MODE_READ,\
                                 &beta_in_view);CHKERRQ(ierr);
    ierr = MatLoad(appctx->beta_in,beta_in_view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&beta_in_view);CHKERRQ(ierr);


    /* Load precomputed rotation matrix */
    PetscViewer    rot_view;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,\
                                 "rot.dat",\
                                 FILE_MODE_READ,\
                                 &rot_view);CHKERRQ(ierr);
    ierr = MatLoad(appctx->rot,rot_view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&rot_view);CHKERRQ(ierr);

  return ierr;
}


/* ---------------------------------------------------------------------
   finalize - finalze the problem.
   AppCtx - user-defined application context
 --------------------------------------------------------------------- */
PetscErrorCode finalize(void *ctx)
{
    PetscErrorCode ierr;
    AppCtx         *appctx = (AppCtx*) ctx;   /* user-defined application context */

    ierr = MatDestroy(&appctx->beta_in); CHKERRQ(ierr);
    ierr = MatDestroy(&appctx->beta_out); CHKERRQ(ierr);
    ierr = MatDestroy(&appctx->rot); CHKERRQ(ierr);
    ierr = VecDestroy(&appctx->X); CHKERRQ(ierr);
    ierr = VecDestroy(&appctx->Y); CHKERRQ(ierr);

  return ierr;
}


/* ---------------------------------------------------------------------
   submatrixtovector - convert (sub)matrix to 1D vector, perform MatMult ! .
   AppCtx - user-defined application context
 --------------------------------------------------------------------- */
PetscErrorCode submatovector(Mat in, Vec work, PetscInt start, char filename[], void *ctx)
{
    PetscErrorCode ierr;
    AppCtx             *appctx = (AppCtx*) ctx;
    Mat                A = in;
    Vec                X = work;
    PetscInt           rowstart = start;
    PetscInt           rowend = start + appctx-> cols;
    PetscInt           matstart,matend;
    PetscInt           row,rowidx[1];
    const PetscScalar  *data;

    ierr = VecSet(X,0); CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(A,&matstart,&matend); CHKERRQ(ierr);

    if ((matstart <= rowstart) || (matend <= rowend)){
	for (row=rowstart; row<rowend; row++){
		rowidx[0] = row;
		ierr = MatGetRow(A,row,NULL,NULL,&data); CHKERRQ(ierr);
		ierr = VecSetValuesBlocked(X,1, rowidx, &data[0], INSERT_VALUES); CHKERRQ(ierr);
		ierr = MatRestoreRow(A,row,NULL,NULL,&data); CHKERRQ(ierr);
		}
	    }

    ierr = VecAssemblyBegin(X); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(X); CHKERRQ(ierr);

    if (appctx->debug_flag){
    ierr = VecSetBlockSize(X,1); CHKERRQ(ierr);
    PetscViewer x;  /* viewer to write the solution to hdf5*/
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,filename,
                               FILE_MODE_WRITE,&x);CHKERRQ(ierr);
    ierr = VecView(X,x);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&x);CHKERRQ(ierr);
    ierr = VecSetBlockSize(appctx->X,appctx->cols); CHKERRQ(ierr);
    }

  return ierr;
}


/* ---------------------------------------------------------------------
   rotate - rotate using discretized rotation matrix
   AppCtx - user-defined application context
 --------------------------------------------------------------------- */
PetscErrorCode rotate(Mat rotation_matrix, Vec input, Vec output, char filename[], void *ctx)
{
    PetscErrorCode ierr;
    AppCtx         *appctx = (AppCtx*) ctx;   /* user-defined application context */
    Mat            rot = rotation_matrix;
    Vec            in  = input;
    Vec            out = output;

    /* Matrix vector multiplication */
    ierr = MatMult(rot,in,out); CHKERRQ(ierr);


    if (appctx->debug_flag){
    ierr = VecSetBlockSize(out,1); CHKERRQ(ierr);
    PetscViewer y_view;  /* viewer to write the solution to hdf5*/
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,filename,
                               FILE_MODE_WRITE,&y_view);CHKERRQ(ierr);
    ierr = VecView(out,y_view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&y_view);CHKERRQ(ierr);
    ierr = VecSetBlockSize(out,appctx->cols); CHKERRQ(ierr);
    }


  return ierr;
}


/* ---------------------------------------------------------------------
   rotate - rotate using discretized rotation matrix
   AppCtx - user-defined application context
 --------------------------------------------------------------------- */
PetscErrorCode vectortosubmatrix(Vec data, Mat dest, PetscInt start, char filename[], void *ctx)
{
    PetscErrorCode ierr;
    AppCtx             *appctx = (AppCtx*) ctx;   /* user-defined application context */
    Vec                vec_in = data;
    Mat                mat_out = dest;
    PetscInt           vecstart,vecend;
    PetscInt           vecrowstart,vecrowend;
    PetscInt           rowstart = start;
    PetscInt           row,col;
    PetscInt           rowidx[1];
    PetscInt           colidx[appctx->cols];
    const PetscScalar  *_data;

    for(col=0; col<appctx->cols; col++){colidx[col] = col;}

    /* convert 1D vector to (sub)matrix*/
    ierr = VecGetOwnershipRange(vec_in,&vecstart,&vecend); CHKERRQ(ierr);
    ierr = VecGetArrayRead(vec_in, &_data); CHKERRQ(ierr);
    vecrowstart = vecstart/appctx->cols;
    vecrowend   = vecend/appctx->cols;

    //PetscPrintf(PETSC_COMM_SELF,"rank: %d, rowstart: %d, rowend: %d\n", appctx->rank,rowstart,rowend);

    for (row=vecrowstart; row<vecrowend; row++){
        rowidx[0] = row + rowstart;
        MatSetValues(mat_out, 1, rowidx, appctx->cols, colidx,\
                     (_data + (row-vecrowstart)*appctx->cols), ADD_VALUES); CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(mat_out,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mat_out,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = VecRestoreArrayRead(vec_in, &_data); CHKERRQ(ierr);

    if (appctx->debug_flag){
    ierr = VecSetBlockSize(vec_in,1); CHKERRQ(ierr);
    PetscViewer y_view;  /* viewer to write the solution to hdf5*/
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,filename,
                               FILE_MODE_WRITE,&y_view);CHKERRQ(ierr);
    ierr = VecView(vec_in,y_view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&y_view);CHKERRQ(ierr);
    ierr = VecSetBlockSize(vec_in,appctx->cols); CHKERRQ(ierr);
    }

  return ierr;
}
