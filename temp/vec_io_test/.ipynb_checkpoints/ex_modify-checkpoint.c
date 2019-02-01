
static char help[] = "Tests I/O of vectors for HDF5\n\n";

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
  PetscScalar const *values;


  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);

  /* PART 1:  Generate vector, then write it in the given data format */
/*
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
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"vector.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(u,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
*/
  /* PART 2:  Read in vector in binary format */

  /* Read new vector in binary format */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"reading vector in hdf5 from vector.dat ...\n");CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"ref_index.h5",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u,"ref_index");CHKERRQ(ierr);
  ierr = VecLoad(u,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecGetArrayRead(u,&values);CHKERRQ(ierr);
  ierr = VecGetLocalSize(u,&ldim);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(u,&low,NULL);CHKERRQ(ierr);
  for (i=0; i<ldim; i++) {
    if (values[i] != (PetscScalar)(i + low)) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Data check failed!\n");
  }
  ierr = VecRestoreArrayRead(u,&values);CHKERRQ(ierr);

  /* Free data structures */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}