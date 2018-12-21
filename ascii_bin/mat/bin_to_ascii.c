#include <stdio.h>

/*  This file takes an input of Petsc readable bianry
 *  matrix and converts it to ascii.

 *  The Petsc format for a vector is a header that consists
 *  of four 4 byte integers MAT_FILE_CLASSID, number of elements.
 *  This is followed by the number of rows and columns of the
 *  matrix itself. The matrix is stored as ieee-754 doubles
 *  in big-endian order.

 *  This code works for complex vectors only. 
 *  If scalar vectors are to be used, the loop end 
 *  criterion has to be changed.
 *  Note that complex numbers are stored as two successive doubles.

 *  This code uses a function that swaps the byte order
 *  (this is needed beacuse x86_64 stores numbers in little 
 *  endian format and petsc uses big endian) from the website
 *  stackoverflow.com

 *  Author : Sajid, Date : 13 December 2018
 */


/*From stackoverflow.com; Converts byte order*/
void SwapBytes(void *pv, size_t n) {
	char *p = pv;  
	size_t lo, hi; 
	for(lo=0, hi=n-1; hi>lo; lo++, hi--){ 
		char tmp=p[lo];
		p[lo] = p[hi];  
		p[hi] = tmp; 
		}
} 

void main(int argc, char* argv[]){
	
	FILE* file_read  = fopen(argv[1],"r");
	FILE* file_write = fopen("result_ascii.dat","w");
	int               i=0; //Loop counter
	int            temp_1; //Read header as ints
	double         temp_2; //Read data as double
	int        num_rows=1; //Loop control variable
	int        num_cols=1; //Loop control variable

	while (i<(num_rows*num_cols*2+4)){
		//First number is MAT_FILE_CLASSID
		if (i==0){
		fread(&temp_1,1,sizeof(temp_1),file_read);
		SwapBytes(&temp_1,sizeof(temp_1));
		i++;}
		//Second number is number of rows
		if (i==1){
		fread(&temp_1,1,sizeof(temp_1),file_read);
		SwapBytes(&temp_1,sizeof(temp_1));
		num_rows = temp_1;
		i++;}
		//Third number is number of cols
		if (i==2){
		fread(&temp_1,1,sizeof(temp_1),file_read);
		SwapBytes(&temp_1,sizeof(temp_1));
		num_cols = temp_1;
		i++;}
		//Fourth number is -1
		if (i==3){
		fread(&temp_1,1,sizeof(temp_1),file_read);
		SwapBytes(&temp_1,sizeof(temp_1));
		i++;}
		//Read the elements
		else{
		fread(&temp_2,1,sizeof(temp_2),file_read);
		SwapBytes(&temp_2,sizeof(temp_2));
		fprintf(file_write,"%lf\n",temp_2);
		i++;}
		}
	fclose(file_read);
	fclose(file_write);
}
