
#include <petscmat.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/

#ifndef STRUCT_APPCTX
#define STRUCT_APPCTX

typedef struct{
	Mat            beta_in;              /* beta matrix*/
	Mat            beta_out;             /* rotated beta matrix*/
	Mat            rot;                  /* rotation matrix*/
	Vec            X;                    /* work vector */
	Vec            Y;                    /* work vector */
	PetscInt       cols;                 /* object size is 64^3 */
	PetscMPIInt    rank,size;
	PetscBool      debug_flag;
	} AppCtx;

#endif


