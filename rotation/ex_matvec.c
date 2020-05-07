static char help[] = "Rotate a 3D object stored as 2D matrix!\n\n";

/* Rotate a 3D object stored as 2D matrix! */

#include <petscmat.h>
#include <petscviewerhdf5.h>
#include "appctx.h"

extern PetscErrorCode initialize(void*);
extern PetscErrorCode finalize(void*);
extern PetscErrorCode submatovector(Mat, Vec, PetscInt, char *, void *);
extern PetscErrorCode rotate(Mat, Vec, Vec, char *, void*);
extern PetscErrorCode vectortosubmatrix(Vec, Mat, PetscInt, char *, void *);

int main(int argc,char **args)
{
    AppCtx           appctx;          /* user-defined application context */
    PetscErrorCode   ierr;
    char fname[50];
    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
    
    initialize(&appctx);
	
    sprintf(fname,"X.h5");
    submatovector(appctx.beta_in, appctx.X, 0, fname, &appctx);

    sprintf(fname,"Y_rot.h5");
    rotate(appctx.rot, appctx.X, appctx.Y, fname, &appctx);

    sprintf(fname,"Y.h5");
    vectortosubmatrix(appctx.Y, appctx.beta_out, 0, fname, &appctx);

    sprintf(fname,"X_rot.h5");
    submatovector(appctx.beta_out, appctx.X, 0, fname, &appctx);
    
    finalize(&appctx);
    
    ierr = PetscFinalize();
    return ierr;
}
