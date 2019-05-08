static char help[] = "2D (MPI) Far field Prop. No output phase correction! \n\n";

#include <petscmat.h>
#include <petscviewerhdf5.h>

/*    User-defined application context - contains data needed 
 *    by the application-provided call-back routines. */
typedef struct {   
  PetscInt    mx;               /* total number of grid points in x */
  PetscInt    my;               /* total number of grid points in y */
  PetscReal   step_grid_x;      /* grid spacing in x */
  PetscReal   step_grid_y;      /* grid spacing in y */ 
  PetscReal   lambda;           /* wavelength */
  PetscReal   energy;           /* energy in ev */
  PetscReal   step_z;           /* step size in time */
  PetscViewer hdf5_sol_viewer;  /* viewer to write the solution to hdf5*/ 
    }AppCtx;

extern PetscErrorCode make1FTvec(Vec,AppCtx*);
extern PetscErrorCode fftshift2D(Vec,Vec);

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  AppCtx         appctx;          /* user-defined application context */
  PetscMPIInt    rank,size;       /*  MPI size and rank */
  Vec            u,u_,H;          /* wave, work and transfer function vectors */
  Mat            A;               /* FFT-matrix to call FFTW via interface */
  PetscReal      prop_distance;   /* propagation distance */
  
  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);

/* ---------------------------------------------------------------------
   Initialize parameters from input!
   --------------------------------------------------------------------- */
  appctx.mx = 8192;
  appctx.my = 8192;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mx",&appctx.mx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-my",&appctx.my,NULL);CHKERRQ(ierr);
 
  appctx.energy  = 12000;
  ierr = PetscOptionsGetReal(NULL,NULL,"-energy",&appctx.energy,NULL);CHKERRQ(ierr);
  appctx.lambda    = (1239.84/appctx.energy)*1e-9;
 
  prop_distance   = 0.0001;
  ierr = PetscOptionsGetReal(NULL,NULL,"-prop_distance",&prop_distance,NULL);CHKERRQ(ierr);
  appctx.step_z = prop_distance;

  appctx.step_grid_x = 1.0987669393246516e-9;
  appctx.step_grid_y = 1.0987669393246516e-9;
  ierr = PetscOptionsGetReal(NULL,NULL,"-step_grid_x",
                             &appctx.step_grid_x,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-step_grid_y",
                             &appctx.step_grid_y,NULL);CHKERRQ(ierr);
 

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Far Field prop on %d processors!\n",size);
	  CHKERRQ(ierr); 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"mx : %d, my: %d, energy(in eV) : %e\n",
		      appctx.mx, appctx.my, appctx.energy);CHKERRQ(ierr);       


/* ---------------------------------------------------------------------
   Make FFT matrix (via interface) and create vecs aligned to it. 
   --------------------------------------------------------------------- */
  PetscInt       dim[2],DIM;             /* FFT parameters */
  PetscScalar    scale;                  /* FFT scaling parameter */  
  dim[0] = appctx.mx;
  dim[1] = appctx.my;
  DIM    = 2;
  ierr   = MatCreateFFT(PETSC_COMM_WORLD,DIM,dim,MATFFTW,&A);CHKERRQ(ierr);
 
  /* Create vectors that are compatible with parallel layout of A - must call MatCreateVecs()! */
  ierr = MatCreateVecsFFTW(A,&u,&u_,NULL);CHKERRQ(ierr);
  Vec shift;
  ierr = MatCreateVecsFFTW(A,NULL,&shift,NULL);CHKERRQ(ierr);

 
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE, appctx.mx*appctx.my,&H);CHKERRQ(ierr);
  /* Initialize the Transfer function to be used*/
  ierr = make1FTvec(H,&appctx);CHKERRQ(ierr);

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Read exit wave   
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE, appctx.mx*appctx.my,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u,"exit_wave");CHKERRQ(ierr);
  PetscViewer wave_viewer;
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"exit_wave.h5",
	       FILE_MODE_READ,&wave_viewer);
         CHKERRQ(ierr);
  ierr = VecLoad(u,wave_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&wave_viewer);CHKERRQ(ierr);
    
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Run the main loop. 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/ 
  scale  = ((1.0*PETSC_i)/(appctx.lambda*appctx.step_z))*(appctx.step_grid_x * appctx.step_grid_y);
      
  ierr = VecPointwiseMult(u,u,H);CHKERRQ(ierr);
  ierr = MatMult(A,u,u_);CHKERRQ(ierr);
  ierr = VecScale(u_,scale);CHKERRQ(ierr);	  

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Done!\n");CHKERRQ(ierr);    
  ierr = PetscBarrier(NULL);CHKERRQ(ierr);

  ierr = fftshift2D(u_,shift);CHKERRQ(ierr);

  /* Write H to hdf5, for debugging.
  PetscViewer hdf5_tf_viewer;
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"Hvec.h5", FILE_MODE_WRITE,&hdf5_tf_viewer);CHKERRQ(ierr);
  ierr = VecView(H,hdf5_tf_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&hdf5_tf_viewer);CHKERRQ(ierr); */

  /* Write shift, now containing the far field wave to hdf5. */
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"final.h5",
		             FILE_MODE_WRITE,&appctx.hdf5_sol_viewer);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)shift,"final");CHKERRQ(ierr);
  ierr = VecView(shift,appctx.hdf5_sol_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&appctx.hdf5_sol_viewer);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&u_);CHKERRQ(ierr);
  ierr = VecDestroy(&H);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}


/* ---------------------------------------------------------------------
   makeTFvec - compute the Transfer Function matrix with the 
               parameters and store it as a vector.
  --------------------------------------------------------------------- */
  

PetscErrorCode make1FTvec(Vec H,AppCtx* appctx){
  PetscErrorCode  ierr;
  PetscInt        i,start,end,row,col;
  PetscReal       X,Y;
  PetscScalar     prefac,v;
  PetscReal       pi = 3.14159265359;
  PetscReal       lambda = appctx->lambda;
  PetscReal       sx = appctx->step_grid_x;
  PetscReal       sy = appctx->step_grid_y;
  PetscReal       z = appctx->step_z;
  PetscInt        N = appctx->mx;
  PetscReal       n = 1/(PetscReal)N;

  ierr = VecGetOwnershipRange(H,&start,&end);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)H, "H_vec");CHKERRQ(ierr);

  prefac = -1*PETSC_i*((2*pi)/(lambda));

  /*For each row i, loop over all cols j*/
  for(i=start;i<end;i++){
	row = i/N;
	col = i - row*N;
	X = (row-N/2)*sx;
	Y = (col-N/2)*sy;
	v = PetscExpComplex(prefac*PetscSqrtReal((X*X + Y*Y + z*z)));
	ierr = VecSetValues(H,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(H);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(H);CHKERRQ(ierr);

  return 0;
}




/* ---------------------------------------------------------------------
   fftshift2D - fftshift a vector, store result in new vector
  --------------------------------------------------------------------- */ 
PetscErrorCode fftshift2D(Vec vec1,Vec vec2){
  PetscErrorCode ierr;
  IS             is1,is2;
  PetscInt       i,j,v,start1,start2,end1,end2;
  PetscInt       N,row,col;
  PetscInt       *indices1;
  PetscInt       *indices2;
  PetscInt       idx_size1;
  PetscInt       idx_size2;
  
  ierr = VecGetSize(vec1,&N);CHKERRQ(ierr);
  N = (PetscInt) PetscSqrtReal(N);
 
  ierr = VecGetLocalSize(vec1,&idx_size1);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec2,&idx_size2);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(vec1,&start1,&end1);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(vec2,&start2,&end2);CHKERRQ(ierr);

  ierr = PetscMalloc1(idx_size1,&indices1);CHKERRQ(ierr);
  ierr = PetscMalloc1(idx_size2,&indices2);CHKERRQ(ierr);

  for(i=0;i<idx_size1;i++){indices1[i] = start1+ i;}

  for(i=0;i<idx_size2;i++){
	  j = start1 + i; 
	  row = j/N;
	  col = j - row*N;
	  if(row < N/2){
		if(col < N/2) {v = (row + N/2)*N + col+N/2;}
		if(col >= N/2){v = (row + N/2)*N + col-N/2;}
	  	}
	  if(row >= N/2){
		if(col < N/2) {v = (row - N/2)*N + col+N/2;}
		if(col >= N/2){v = (row - N/2)*N + col-N/2;}
	  	}
	 indices2[i] = v;
  }
  ierr = PetscBarrier(NULL);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,idx_size1,indices1,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,idx_size2,indices2,PETSC_COPY_VALUES,&is2);CHKERRQ(ierr);

  ierr = PetscFree(indices1);CHKERRQ(ierr);
  ierr = PetscFree(indices2);CHKERRQ(ierr);
 

  VecScatter fftshift;
  ierr = VecScatterCreate(vec1,is1,vec2,is2,&fftshift);CHKERRQ(ierr);
  ierr = VecScatterBegin(fftshift,vec1,vec2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(fftshift,vec1,vec2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);

  return 0;
}
