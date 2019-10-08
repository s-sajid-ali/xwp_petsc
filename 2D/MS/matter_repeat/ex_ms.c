static char help[] = "2D Multislice,  MPI-multislice, matter-repeat \n\n";

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

extern PetscErrorCode makeTFvec(Vec,AppCtx*);
extern PetscErrorCode makeinput(Vec,AppCtx*);
extern PetscErrorCode ms_loop(Mat,Vec,Vec,Vec,Vec,PetscInt,AppCtx*);


int main(int argc,char **args)
{
  PetscErrorCode ierr;
  AppCtx         appctx;          /* user-defined application context */
  PetscMPIInt    rank,size;       /*  MPI size and rank */
  Vec            u,u_,H;          /* wave, work and transfer function vectors */
  Vec            slice_rid;       /* vector to hold the refractive index */ 
  Mat            A;               /* FFT-matrix to call FFTW via interface */
  PetscInt       prop_steps;      /* number of steps for propagation */
  PetscReal      prop_distance;   /* propagation distance */
  PetscReal      delta,beta;      /* delta, beta -> ref index */
  PetscScalar    rid_fac;         /* multiplicative factor */
  PetscReal      pi = 3.14159265359;


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
 
  prop_distance   = 5e-6;
  ierr = PetscOptionsGetReal(NULL,NULL,"-prop_distance",&prop_distance,NULL);CHKERRQ(ierr);

  prop_steps      = 25;
  ierr= PetscOptionsGetInt(NULL,NULL,"-prop_steps",&prop_steps,NULL);CHKERRQ(ierr);

  appctx.step_grid_x = 1.0987669393246516e-09;
  appctx.step_grid_y = 1.0987669393246516e-09;
  ierr = PetscOptionsGetReal(NULL,NULL,"-step_grid_x",
                             &appctx.step_grid_x,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-step_grid_y",
                             &appctx.step_grid_y,NULL);CHKERRQ(ierr);

  delta = 1.8402e-05;
  beta  = 2.8532e-06;

  appctx.step_z = prop_distance/prop_steps;
  
  rid_fac       = 2*pi*(appctx.step_z/appctx.lambda)*(PETSC_i*delta-beta);


  ierr = PetscPrintf(PETSC_COMM_WORLD,"Multi-slice xwp on %d processors!\n",size);
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
  scale  = 1.0/(PetscReal)(appctx.mx * appctx.my);
  ierr   = MatCreateFFT(PETSC_COMM_WORLD,DIM,dim,MATFFTW,&A);CHKERRQ(ierr);
 
  /* Create vectors that are compatible with parallel layout of A - must call MatCreateVecs()! */
  ierr = MatCreateVecsFFTW(A,&u,&u_,&H);CHKERRQ(ierr);
  //ierr = MatCreateVecsFFTW(A,&slice_rid,NULL,NULL);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE, appctx.mx*appctx.my,&slice_rid);CHKERRQ(ierr);
  //ierr = VecDuplicate(H,&slice_rid);
  
  /* Initialize the Transfer function to be used*/
  ierr = makeTFvec(H,&appctx);CHKERRQ(ierr);

  /* Initialize the input vector */ 
  ierr = makeinput(u,&appctx);CHKERRQ(ierr);


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create a vector to hold refractive index at slice_rid. Destroy the viewer   
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
  PetscViewer hdf_ref_index_viewer;
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"ref_index_ms.h5",
	       FILE_MODE_READ,&hdf_ref_index_viewer);
         CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)slice_rid,"ref_index");CHKERRQ(ierr);
  ierr = VecLoad(slice_rid,hdf_ref_index_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&hdf_ref_index_viewer);CHKERRQ(ierr);
  ierr = VecScale(slice_rid,rid_fac);CHKERRQ(ierr);
  ierr = VecExp(slice_rid);CHKERRQ(ierr);

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Run the main loop. 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/ 
  ierr = ms_loop(A,slice_rid,u, u_,H,prop_steps,&appctx);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Done!\n");CHKERRQ(ierr);    


  /* Write H to hdf5, for debugging. 
  PetscViewer hdf5_tf_viewer;
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"Hvec.h5", FILE_MODE_WRITE,&hdf5_tf_viewer);CHKERRQ(ierr);
  ierr = VecView(H,hdf5_tf_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&hdf5_tf_viewer);CHKERRQ(ierr);*/

  /* Write u, now containing the exit wave to hdf5. */
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"exit_wave.h5",
		             FILE_MODE_WRITE,&appctx.hdf5_sol_viewer);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u, "exit_wave");CHKERRQ(ierr);
  ierr = VecView(u,appctx.hdf5_sol_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&appctx.hdf5_sol_viewer);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&slice_rid);CHKERRQ(ierr);
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
  

PetscErrorCode makeTFvec(Vec H,AppCtx* appctx){
  PetscErrorCode  ierr;
  PetscInt        i,start,end,row,col;
  PetscReal       Fx,Fy;
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

  prefac = -1*PETSC_i*(2*pi*z/lambda);
  /*For each row i, loop over all cols j*/
  for(i=start;i<end;i++){
	row = i/N;
	col = i - row*N;
	if(row < N/2) {Fx = (row)     * n * 1/sx;}
 	if(row >= N/2){Fx = (row - N) * n * 1/sx;}
 	if(col < N/2) {Fy = (col)     * n * 1/sy;}
	if(col >= N/2){Fy = (col - N) * n * 1/sy;}
	v = PetscExpComplex( prefac * PetscSqrtComplex(1 - lambda*lambda*(Fx*Fx + Fy*Fy)));
	//v = PetscSqrtComplex(1 - lambda*lambda*(Fx*Fx+Fy*Fy));
	ierr = VecSetValues(H,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(H);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(H);CHKERRQ(ierr);

  return 0;
}


/* ---------------------------------------------------------------------
   makeinput - initialize u vector with the input wave.
  --------------------------------------------------------------------- */

PetscErrorCode makeinput(Vec u,AppCtx* appctx){
  PetscErrorCode  ierr;
  PetscInt        i,start,end,row,col;
  PetscInt        mx = appctx->mx;
  PetscInt        my = appctx->my;
  PetscInt        N = mx;
  PetscScalar     v;


  ierr = VecGetOwnershipRange(u,&start,&end);CHKERRQ(ierr);
  
  v = 1;
  ierr = VecSet(u,v); CHKERRQ(ierr);

  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);

  return 0;
}


/* ---------------------------------------------------------------------
   Main multi-slice loop!
   2D-FFT, Point-wise multiply with transfer function, 2D-IFFT.
   Scaling to account for FFTW convention.
   --------------------------------------------------------------------- */
 
PetscErrorCode ms_loop(Mat A, Vec slice_rid, Vec u, Vec u_, Vec H, PetscInt prop_steps,AppCtx* appctx){
  PetscErrorCode  ierr;
  PetscInt        i;               /* counter for multi-slice loop */
  PetscInt        mx = appctx->mx;
  PetscInt        my = appctx->my;
  PetscScalar     scale = 1/(PetscReal)(mx*my);


  for(i=0; i<prop_steps; i++){
	  ierr = PetscPrintf(PETSC_COMM_WORLD,"Slice Num : %d \n",i);CHKERRQ(ierr);    
	  ierr = MatMult(A,u,u_);CHKERRQ(ierr);
	  ierr = VecPointwiseMult(u_,u_,H);CHKERRQ(ierr);
	  ierr = MatMultTranspose(A,u_,u);CHKERRQ(ierr);
	  ierr = VecScale(u,scale);CHKERRQ(ierr);
	  ierr = VecPointwiseMult(u,u,slice_rid);CHKERRQ(ierr);
	  }

  return 0;
}
