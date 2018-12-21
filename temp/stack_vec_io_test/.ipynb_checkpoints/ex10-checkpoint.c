
static char help[] = "Tests I/O of vectors for different data formats (binary,HDF5) and illustrates the use of user-defined event logging\n\n";

#include <petscvec.h>
#include <petscviewerhdf5.h>

/* Note:  Most applications would not read and write a vector within
  the same program.  This example is intended only to demonstrate
  both input and output and is written for use with either 1,2,or 4 processors. */

int main(int argc,char **args)
{
  PetscErrorCode    ierr;
  PetscMPIInt       rank,size;
  PetscInt          i,m = 20,low,high,ldim,iglobal,lsize;
  PetscScalar       v;
  Vec               u;
  PetscViewer       viewer;
  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);

  /* PART 1:  Generate vector, then write it in the given data format */

 // Generate vector 
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u, "Test_Vec");CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(u,&low,&high);CHKERRQ(ierr);
  ierr = VecGetLocalSize(u,&ldim);CHKERRQ(ierr);
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v       = (PetscScalar)(i + low);
    ierr    = VecSetValues(u,1,&iglobal,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
  ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"writing vector in hdf5 to vector.dat ...\n");CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"vector.h5",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5SetTimestep(viewer, 0);CHKERRQ(ierr);
  ierr = VecView(u,viewer);CHKERRQ(ierr);
  
  ierr = PetscViewerHDF5SetTimestep(viewer, 1);CHKERRQ(ierr);
  ierr = VecView(u,viewer);CHKERRQ(ierr);
 
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /* Free data structures */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     test:
       nsize: 2
       args: -binary

     test:
       suffix: 2
       nsize: 3
       args: -binary

     test:
       suffix: 3
       nsize: 5
       args: -binary

     test:
       suffix: 4
       requires: hdf5
       nsize: 2
       args: -hdf5

     test:
       suffix: 5
       nsize: 4
       args: -binary -sizes_set

     test:
       suffix: 6
       requires: hdf5
       nsize: 4
       args: -hdf5 -sizes_set


TEST*/
