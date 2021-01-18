#include <chrono>
#include <ctime>
#include <iostream>
#include <string>
#include <cstdlib>
#include <math.h>
#include <stdio.h>


using namespace std;

//Global Constants
const float TOLERANCE = 1.0e-5f;
const float ERROR_INIT = TOLERANCE+1;
const int ITER_MAX = 1000;

const int RANGE = 100;

const int DIMXJ = 2048;
const int DIMYJ = 2082;

const int DIMX = 1888;
const int DIMY = 1888;
const int DIMZ = 1888;

const bool PRINT = true;
const bool PRINTM = false;
const int  LOOPS = 5;

/**
 * Fill a Matrix with random numbers and sparsity
 */
void fillRandomMatrixFloat(float* m, int height, int width, int range) {
	for (int r = 0; r < height; r++)
		for (int c = 0; c < width; c++) {
			m[r * width + c] = rand() % range;
		}
}
void fillRandomMatrixFloat(float *matrix) {
	fillRandomMatrixFloat(matrix, DIMYJ, DIMXJ, RANGE);
}

void copyTmpMatrix(float* in, float* tmp) {
	for(int i = 0; i < DIMXJ*DIMYJ; i++) {
		tmp[i] = in[i];
	}
}

/**
 * Print given Matrix
 */
void printMatrixFloat(float* matrix, int height, int width, string description) {
	cout << "\nMatrix: " << description << "\n";

	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width; c++) {
			cout << matrix[r * width + c] << "\t";
		}
		cout << "\n";
	}
}
void printMatrixFloat(float *matrix, string description) {
	printMatrixFloat(matrix, DIMYJ, DIMXJ, description);
}

//-----------Jacobi Methods------------
/**
 * Sequential Calculation
 * @param A
 */
float jacobi_sequential(float* A) {
	float error = ERROR_INIT;
	int iter = 0;

	float* Anew = new float[DIMXJ*DIMYJ];

	printf("Start Jacobi Method sequentially.\n");

	// convergence loop
	while ( error > TOLERANCE && iter < ITER_MAX)  {
		error = 0.0;

		// calculation for each element
		for( int j = 1; j < DIMXJ-1; j++) {
			for( int i = 1; i < DIMYJ-1; i++ ) {
				*(Anew+(j*DIMXJ+i)) = 0.25 * ( *(A+ j*DIMYJ+i+1) + *(A + j*DIMXJ+i-1)
				                              + *(A + (j-1)*DIMXJ+i) + *(A + (j+1)*DIMXJ+i));
				error = fmax( error, fabs( *(Anew + j*DIMXJ+i) - *(A + j*DIMXJ+i) ));
			}
		}

		// copy results for returning
		for( int j = 1; j < DIMXJ-1; j++) {
			for( int i = 1; i < DIMYJ-1; i++ ) {
				*(A + j*DIMXJ+i) = *(Anew + j*DIMXJ+i);
			}
		}

		//Inform about intermediate results
		if (PRINT) if(iter % 100 == 0) printf("Iteration: %5d, current error: %0.6f\n", iter, error);

		iter++;
	}

	return error;
}

/**
 * Simple Kernels Calculation
 * @param A
 */
float jacobi_kernels1(float* A) {
	float error = ERROR_INIT;
	int iter = 0;

	float* Anew = new float[DIMXJ*DIMYJ];

	printf("Start Jacobi Method with naive kernels usage.\n");

	// convergence loop
	while ( error > TOLERANCE && iter < ITER_MAX)  {
		error = 0.0;

		// calculation for each element
#pragma acc kernels
		{
			for( int j = 1; j < DIMXJ-1; j++) {
				for( int i = 1; i < DIMYJ-1; i++ ) {
					*(Anew+(j*DIMXJ+i)) = 0.25 * ( *(A+ j*DIMYJ+i+1) + *(A + j*DIMXJ+i-1)
					                              + *(A + (j-1)*DIMXJ+i) + *(A + (j+1)*DIMXJ+i));
					error = fmax( error, fabs( *(Anew + j*DIMXJ+i) - *(A + j*DIMXJ+i) ));
				}
			}

			// copy results for returning
			for( int j = 1; j < DIMXJ-1; j++) {
				for( int i = 1; i < DIMYJ-1; i++ ) {
					*(A + j*DIMXJ+i) = *(Anew + j*DIMXJ+i);
				}
			}
		}

		//Inform about intermediate results
		if (PRINT) if(iter % 100 == 0) printf("Iteration: %5d, current error: %0.6f\n", iter, error);

		iter++;
	}

	return error;
}

/**
 * Kernels Calculation with optimised data usage
 * @param A
 */
float jacobi_kernels2(float* A) {
	float error = ERROR_INIT;
	int iter = 0;

	float* Anew = new float[DIMXJ*DIMYJ];

	printf("Start Jacobi Method with dataoptimised kernels usage.\n");

	// convergence loop
#pragma acc data pcopy(A) create (Anew)
	{
		while ( error > TOLERANCE && iter < ITER_MAX)  {
			error = 0.0;

			// calculation for each element
#pragma acc kernels
			{
				for( int j = 1; j < DIMXJ-1; j++) {
					for( int i = 1; i < DIMYJ-1; i++ ) {
						*(Anew+(j*DIMXJ+i)) = 0.25 * ( *(A+ j*DIMYJ+i+1) + *(A + j*DIMXJ+i-1)
						                              + *(A + (j-1)*DIMXJ+i) + *(A + (j+1)*DIMXJ+i));
						error = fmax( error, fabs( *(Anew + j*DIMXJ+i) - *(A + j*DIMXJ+i) ));
					}
				}

				// copy results for returning
				for( int j = 1; j < DIMXJ-1; j++) {
					for( int i = 1; i < DIMYJ-1; i++ ) {
						*(A + j*DIMXJ+i) = *(Anew + j*DIMXJ+i);
					}
				}
			}

			//Inform about intermediate results
			if (PRINT) if(iter % 100 == 0) printf("Iteration: %5d, current error: %0.6f\n", iter, error);

			iter++;
		}
	}

	return error;
}

/**
 * Kernels Calculation with optimised data usage and collapse
 * @param A
 */
float jacobi_kernels3(float* A) {
	float error = ERROR_INIT;
	int iter = 0;

	float* Anew = new float[DIMXJ*DIMYJ];

	printf("Start Jacobi Method with dataoptimised kernels usage and collapse.\n");

	// convergence loop
#pragma acc data pcopy(A) create (Anew)
	{
		while ( error > TOLERANCE && iter < ITER_MAX)  {
			error = 0.0;

			// calculation for each element
#pragma acc kernels
			{
#pragma acc loop collapse(2)
				for( int j = 1; j < DIMXJ-1; j++) {
					for( int i = 1; i < DIMYJ-1; i++ ) {
						*(Anew+(j*DIMXJ+i)) = 0.25 * ( *(A+ j*DIMYJ+i+1) + *(A + j*DIMXJ+i-1)
						                              + *(A + (j-1)*DIMXJ+i) + *(A + (j+1)*DIMXJ+i));
						error = fmax( error, fabs( *(Anew + j*DIMXJ+i) - *(A + j*DIMXJ+i) ));
					}
				}

				// copy results for returning
#pragma acc loop collapse(2)
				for( int j = 1; j < DIMXJ-1; j++) {
					for( int i = 1; i < DIMYJ-1; i++ ) {
						*(A + j*DIMXJ+i) = *(Anew + j*DIMXJ+i);
					}
				}
			}

			//Inform about intermediate results
			if (PRINT) if(iter % 100 == 0) printf("Iteration: %5d, current error: %0.6f\n", iter, error);

			iter++;
		}
	}

	return error;
}

/**
 * Naive Parallel Calculation
 * @param A
 */
float jacobi_parallel1(float* A) {
	float error = ERROR_INIT;
	int iter = 0;

	float* Anew = new float[DIMXJ*DIMYJ];

	printf("Start Jacobi Method naively parallel.\n");

	// convergence loop
	while ( error > TOLERANCE && iter < ITER_MAX)  {
		error = 0.0;

		// calculation for each element
#pragma acc parallel loop
		for( int j = 1; j < DIMXJ-1; j++) {
#pragma acc loop
			for( int i = 1; i < DIMYJ-1; i++ ) {
				*(Anew+(j*DIMXJ+i)) = 0.25 * ( *(A+ j*DIMYJ+i+1) + *(A + j*DIMXJ+i-1)
				                              + *(A + (j-1)*DIMXJ+i) + *(A + (j+1)*DIMXJ+i));
				error = fmax( error, fabs( *(Anew + j*DIMXJ+i) - *(A + j*DIMXJ+i) ));
			}
		}

		// copy results for returning
#pragma acc parallel loop
		for( int j = 1; j < DIMXJ-1; j++) {
#pragma acc loop
			for( int i = 1; i < DIMYJ-1; i++ ) {
				*(A + j*DIMXJ+i) = *(Anew + j*DIMXJ+i);
			}
		}

		//Inform about intermediate results
		if (PRINT) printf("Iteration: %5d, current error: %0.6f\n", iter, error);

		iter++;
	}

	return error;
}

/**
 * Dataoptimised Parallel Calculation
 * @param A
 */
float jacobi_parallel2(float* A) {
	float error = ERROR_INIT;
	int iter = 0;

	float* Anew = new float[DIMXJ*DIMYJ];

	printf("Start Jacobi Method parallel with data region.\n");

	// convergence loop
#pragma acc data pcopy(A) create (Anew)
	{
		while ( error > TOLERANCE && iter < ITER_MAX)  {
			error = 0.0;

			// calculation for each element
#pragma acc parallel loop
			for( int j = 1; j < DIMXJ-1; j++) {
#pragma acc loop
				for( int i = 1; i < DIMYJ-1; i++ ) {
					*(Anew+(j*DIMXJ+i)) = 0.25 * ( *(A+ j*DIMYJ+i+1) + *(A + j*DIMXJ+i-1)
					                              + *(A + (j-1)*DIMXJ+i) + *(A + (j+1)*DIMXJ+i));
					error = fmax( error, fabs( *(Anew + j*DIMXJ+i) - *(A + j*DIMXJ+i) ));
				}
			}

			// copy results for returning
#pragma acc parallel loop
			for( int j = 1; j < DIMXJ-1; j++) {
#pragma acc loop
				for( int i = 1; i < DIMYJ-1; i++ ) {
					*(A + j*DIMXJ+i) = *(Anew + j*DIMXJ+i);
				}
			}

			//Inform about intermediate results
			if (PRINT) printf("Iteration: %5d, current error: %0.6f\n", iter, error);

			iter++;
		}
	}

	return error;
}

/**
 * Dataoptimised Parallel Calculation with asynchronous calculatoion
 * @param A
 */
float jacobi_parallel3(float* A) {
	float error = ERROR_INIT;
	int iter = 0;

	float* Anew = new float[DIMXJ*DIMYJ];

	printf("Start Jacobi Method parallel asynchronous with data region.\n");

	// convergence loop
#pragma acc data pcopy(A) create(Anew)
	{
		while ( abs(error) > TOLERANCE && iter < ITER_MAX)  {
			error = 0.0;

			// calculation for each element
#pragma acc parallel loop async(1) reduction(max:error)
			for( int j = 1; j < DIMXJ-1; j++) {
#pragma acc loop reduction(max:error)
				for( int i = 1; i < DIMYJ-1; i++ ) {
					*(Anew+(j*DIMXJ+i)) = 0.25 * ( *(A+ j*DIMYJ+i+1) + *(A + j*DIMXJ+i-1)
					                              + *(A + (j-1)*DIMXJ+i) + *(A + (j+1)*DIMXJ+i));
					error = *(Anew + j*DIMXJ+i) - *(A + j*DIMXJ+i);
				}
			}

			// copy results for returning
#pragma acc wait(1) async(2)
#pragma acc parallel loop async(2)
			for( int j = 1; j < DIMXJ-1; j++) {
#pragma acc loop
				for( int i = 1; i < DIMYJ-1; i++ ) {
					*(A + j*DIMXJ+i) = *(Anew + j*DIMXJ+i);
				}
			}

			//Inform about intermediate results
			if (PRINT) printf("Iteration: %5d, current error: %0.6f\n", iter, error);

			iter++;

#pragma acc update self(A) async(2)
#pragma acc wait(2)
		}
	}

	return error;
}

/**
 * Dataoptimised Parallel Calculation with collapse
 * @param A
 */
float jacobi_parallel4(float* A) {
	float error = ERROR_INIT;
	int iter = 0;

	float* Anew = new float[DIMXJ*DIMYJ];

	printf("Start Jacobi Method parallel with data region and collapsing loops.\n");

	// convergence loop
#pragma acc data pcopy(A) create (Anew)
	{
		while ( error > TOLERANCE && iter < ITER_MAX)  {
			error = 0.0;

			// calculation for each element
#pragma acc parallel loop collapse(2)
			for( int j = 1; j < DIMXJ-1; j++) {
				//#pragma acc loop
				for( int i = 1; i < DIMYJ-1; i++ ) {
					*(Anew+(j*DIMXJ+i)) = 0.25 * ( *(A+ j*DIMYJ+i+1) + *(A + j*DIMXJ+i-1)
					                              + *(A + (j-1)*DIMXJ+i) + *(A + (j+1)*DIMXJ+i));
					error = fmax( error, fabs( *(Anew + j*DIMXJ+i) - *(A + j*DIMXJ+i) ));
				}
			}

			// copy results for returning
#pragma acc parallel loop collapse(2)
			for( int j = 1; j < DIMXJ-1; j++) {
				//#pragma acc loop
				for( int i = 1; i < DIMYJ-1; i++ ) {
					*(A + j*DIMXJ+i) = *(Anew + j*DIMXJ+i);
				}
			}

			//Inform about intermediate results
			if (PRINT) printf("Iteration: %5d, current error: %0.6f\n", iter, error);

			iter++;
		}
	}

	return error;
}

/**
 * Dataoptimised Parallel Calculation with collapse
 * @param A
 */
float jacobi_parallel5(float* A) {
	float error = ERROR_INIT;
	int iter = 0;

	float* Anew = new float[DIMXJ*DIMYJ];

	printf("Start Jacobi Method parallel with data region and collapsing loops.\n");

	// convergence loop
#pragma acc data pcopy(A) create (Anew)
	{
		while ( error > TOLERANCE && iter < ITER_MAX)  {
			error = 0.0;

			// calculation for each element
#pragma acc parallel loop collapse(2)
			for( int j = 1; j < DIMXJ-1; j++) {
				//#pragma acc loop
				for( int i = 1; i < DIMYJ-1; i++ ) {
					Anew[j*DIMXJ+i] = 0.25 * ( A[j*DIMYJ+i+1] + A[j*DIMXJ+i-1]
					                              + A[(j-1)*DIMXJ+i] + A[(j+1)*DIMXJ+i] );
					error = fmax( error, fabs( Anew[j*DIMXJ+i] - A[j*DIMXJ+i] ));
				}
			}

			// copy results for returning
#pragma acc parallel loop collapse(2)
			for( int j = 1; j < DIMXJ-1; j++) {
				//#pragma acc loop
				for( int i = 1; i < DIMYJ-1; i++ ) {
					A [j*DIMXJ+i] = Anew[j*DIMXJ+i];
				}
			}

			//Inform about intermediate results
			if (PRINT) printf("Iteration: %5d, current error: %0.6f\n", iter, error);

			iter++;
		}
	}

	return error;
}

/**
 * Fill a Matrix with random numbers and sparsity
 */
void fillRandomMatrixLong(long* m, int height, int width, int range) {
	for (int r = 0; r < height; r++)
		for (int c = 0; c < width; c++) {
			m[r * width + c] = rand() % range;
		}
}
void fillRandomMatrixLong(long *matrix) {
	fillRandomMatrixLong(matrix, DIMY, DIMX, RANGE);
}

/**
 * Print given Matrix
 */
void printMatrixLong(long *matrix, int height, int width, string description) {
	cout << "Matrix: " << description << "\n";

	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width; c++) {
			cout << matrix[r * width + c] << "\t";
		}
		cout << "\n";
	}
	cout << "\n";
}
void printMatrixLong(long *matrix, string description) {
	printMatrixLong(matrix, DIMY, DIMX, description);
}



//-----------MatMult Methods------------

/**
 * Multiplies two matrices A and B seuqentially
 */
void matmult_sequential(long* A, long* B, long* R) {
	int widthR  = DIMZ;
	int heightR = DIMY;
	printf("Start sequential matrix multiplication.\n");
	//row = row of result matrix, comes from matrixA
	//col = column of result matrix, comes from matrixB
	for (int row = 0; row < heightR; row++) {
		for (int col = 0; col < widthR; col++) {
			//initialise result cell
			R[row * widthR + col] = 0;

			//walk through all cells of a row in matrixA and every cell in a column of matrixB and multiply
			for (int i = 0; i < DIMX; i++) { //widthA == heightB
				R[row * widthR + col] += A[row * DIMX + i] * B[i * DIMZ + col];
			}
		}
	}
}

/**
 * Multiplies two matrices A and B with naive kernels
 */
void matmult_kernels1(long* A, long* B, long* R) {
	int widthR  = DIMZ;
	int heightR = DIMY;
	printf("Start naive kernels matrix multiplication.\n");
	//row = row of result matrix, comes from matrixA
	//col = column of result matrix, comes from matrixB
#pragma acc kernels
	{
		for (int row = 0; row < heightR; row++) {
			for (int col = 0; col < widthR; col++) {
				//initialise result cell
				R[row * widthR + col] = 0;

				//walk through all cells of a row in matrixA and every cell in a column of matrixB and multiply
				for (int i = 0; i < DIMX; i++) { //widthA == heightB
					R[row * widthR + col] += A[row * DIMX + i] * B[i * DIMZ + col];
				}
			}
		}
	}
}

/**
 * Multiplies two matrices A and B with naive kernels but optimised loops
 */
void matmult_kernels1b(long* A, long* B, long* R) {
	int widthR  = DIMZ;
	int heightR = DIMY;
	printf("Start naive kernels matrix multiplication and optimised loops.\n");
	//row = row of result matrix, comes from matrixA
	//col = column of result matrix, comes from matrixB
#pragma acc kernels
	{
		for (int row = 0; row < heightR; row++) {
			for (int col = 0; col < widthR; col++) {
				//initialise result cell
				R[row * widthR + col] = 0;
			}
		}

		for (int row = 0; row < heightR; row++) {
			for (int col = 0; col < widthR; col++) {
				//walk through all cells of a row in matrixA and every cell in a column of matrixB and multiply
				for (int i = 0; i < DIMX; i++) { //widthA == heightB
					R[row * widthR + col] += A[row * DIMX + i] * B[i * DIMZ + col];
				}
			}
		}
	}
}

/**
 * Multiplies two matrices A and B with naive kernels but optimised loops and data optimised
 */
void matmult_kernels2(long* A, long* B, long* R) {
	int widthR  = DIMZ;
	int heightR = DIMY;
	printf("Start naive kernels matrix multiplication and optimised loops.\n");
	//row = row of result matrix, comes from matrixA
	//col = column of result matrix, comes from matrixB
#pragma acc data pcopy(A, B) create (R)
	{
#pragma acc kernels
		{
			for (int row = 0; row < heightR; row++) {
				for (int col = 0; col < widthR; col++) {
					//initialise result cell
					R[row * widthR + col] = 0;
				}
			}

			for (int row = 0; row < heightR; row++) {
				for (int col = 0; col < widthR; col++) {
					//walk through all cells of a row in matrixA and every cell in a column of matrixB and multiply
					for (int i = 0; i < DIMX; i++) { //widthA == heightB
						R[row * widthR + col] += A[row * DIMX + i] * B[i * DIMZ + col];
					}
				}
			}
		}
	}
}

/**
 * Multiplies two matrices A and B with naive kernels but optimised loops and data optimised
 */
void matmult_kernels3(long* A, long* B, long* R) {
	int widthR  = DIMZ;
	int heightR = DIMY;
	printf("Start naive kernels matrix multiplication and optimised data and collapse.\n");
	//row = row of result matrix, comes from matrixA
	//col = column of result matrix, comes from matrixB
#pragma acc data pcopy(A, B) create (R)
#pragma acc kernels
	{
#pragma acc loop collapse(2)
		for (int row = 0; row < heightR; row++) {
			for (int col = 0; col < widthR; col++) {
				//initialise result cell
				R[row * widthR + col] = 0;
			}
		}

#pragma acc loop collapse(3)
		for (int row = 0; row < heightR; row++) {
			for (int col = 0; col < widthR; col++) {
				//walk through all cells of a row in matrixA and every cell in a column of matrixB and multiply
				for (int i = 0; i < DIMX; i++) { //widthA == heightB
					R[row * widthR + col] += A[row * DIMX + i] * B[i * DIMZ + col];
				}
			}
		}
	}
}

/**
 * Multiplies two matrices A and B with naive parallel but optimised loops
 */
void matmult_parallel1(long* A, long* B, long* R) {
	int widthR  = DIMZ;
	int heightR = DIMY;
	printf("Start naive parallel matrix multiplication.\n");
#pragma acc parallel loop
	for (int row = 0; row < heightR; row++) {
#pragma acc loop
		for (int col = 0; col < widthR; col++) {
			//initialise result cell
			R[row * widthR + col] = 0;
		}
	}

#pragma acc parallel loop
	for (int row = 0; row < heightR; row++) {
#pragma acc loop
		for (int col = 0; col < widthR; col++) {
#pragma acc loop
			//walk through all cells of a row in matrixA and every cell in a column of matrixB and multiply
			for (int i = 0; i < DIMX; i++) { //widthA == heightB
				R[row * widthR + col] += A[row * DIMX + i] * B[i * DIMZ + col];
			}
		}
	}
}

/**
 * Multiplies two matrices A and B with parallel and optimised loops and data optimised
 */
void matmult_parallel2(long* A, long* B, long* R) {
	int widthR  = DIMZ;
	int heightR = DIMY;
	printf("Start parallel matrix multiplication with optimised data.\n");
#pragma acc data pcopy(A, B) create (R)
	{
#pragma acc parallel loop
		for (int row = 0; row < heightR; row++) {
#pragma acc loop
			for (int col = 0; col < widthR; col++) {
				//initialise result cell
				R[row * widthR + col] = 0;
			}
		}

#pragma acc parallel loop
		for (int row = 0; row < heightR; row++) {
#pragma acc loop
			for (int col = 0; col < widthR; col++) {
#pragma acc loop
				//walk through all cells of a row in matrixA and every cell in a column of matrixB and multiply
				for (int i = 0; i < DIMX; i++) { //widthA == heightB
					R[row * widthR + col] += A[row * DIMX + i] * B[i * DIMZ + col];
				}
			}
		}
	}
}

/**
 * Multiplies two matrices A and B with parallel and optimised loops and data optimised and async
 */
void matmult_parallel3(long* A, long* B, long* R) {
	int widthR  = DIMZ;
	int heightR = DIMY;
	printf("Start parallel matrix multiplication with opt data + async.\n");
#pragma acc data pcopy(A, B) create (R)
	{
#pragma acc parallel loop async(1)
		for (int row = 0; row < heightR; row++) {
#pragma acc loop
			for (int col = 0; col < widthR; col++) {
				//initialise result cell
				R[row * widthR + col] = 0;
			}
		}

#pragma acc parallel loop async(1)
		for (int row = 0; row < heightR; row++) {
#pragma acc loop
			for (int col = 0; col < widthR; col++) {
#pragma acc loop
				//walk through all cells of a row in matrixA and every cell in a column of matrixB and multiply
				for (int i = 0; i < DIMX; i++) { //widthA == heightB
					R[row * widthR + col] += A[row * DIMX + i] * B[i * DIMZ + col];
				}
			}
		}
	}
#pragma acc update self(R) async(1)
#pragma acc wait(1)
}

/**
 * Multiplies two matrices A and B with parallel and optimised loops and data optimised and collapse
 */
void matmult_parallel4(long* A, long* B, long* R) {
	int widthR  = DIMZ;
	int heightR = DIMY;
	printf("Start parallel matrix multiplication with opt data + collapse.\n");
#pragma acc data pcopy(A, B) create (R)
	{
#pragma acc parallel loop collapse(2)
		for (int row = 0; row < heightR; row++) {
			for (int col = 0; col < widthR; col++) {
				//initialise result cell
				R[row * widthR + col] = 0;
			}
		}

#pragma acc parallel loop collapse(3)
		for (int row = 0; row < heightR; row++) {
			for (int col = 0; col < widthR; col++) {
				//walk through all cells of a row in matrixA and every cell in a column of matrixB and multiply
				for (int i = 0; i < DIMX; i++) { //widthA == heightB
					R[row * widthR + col] += A[row * DIMX + i] * B[i * DIMZ + col];
				}
			}
		}
	}
}

//-----------main Method------------
int main(int argc, char** argv) {

	//Jacobi vars
	float* AJ    = new float[DIMXJ*DIMYJ];
	float* tmpA = new float[DIMXJ*DIMYJ];
	int noOptsJ = 8;
	chrono::duration<double> timetotalj[noOptsJ]; //sequential, kernel1, kernel2, parallel1, parallel2, parallel3, parallel
	float error = 0.0;
	float errortotal[noOptsJ];
	int optimisationcounterj = 0;

	//Matmult vars
	long *A = new long[DIMX * DIMY];
	long *B = new long[DIMZ * DIMX];
	long *R = new long[DIMZ * DIMY];
	int noOpts = 9;
	chrono::duration<double> timetotal[noOpts]; //sequential, kernels1
	int optimisationcounter = 0;

	//both
	srand(time(NULL));
	chrono::high_resolution_clock::time_point start, end;
	chrono::duration<double> time;


	for (int i = 0; i < LOOPS; i++) {
		//--------------------------
		// Matmult

//		printf("-- Time measuring round %d\n", i);
//
//		//init A
//		fillRandomMatrixLong(A);
//		printf("Matrix A initialised\n");
//		if (PRINTM) printMatrixLong(A, "A");
//
//		//init B
//		fillRandomMatrixLong(B);
//		printf("Matrix B initialised\n");
//		if (PRINTM) printMatrixLong(B, "B");
//
//		//Matmult Algorithm:
//		//Sequential
//		start = chrono::high_resolution_clock::now();
//		matmult_sequential(A, B, R);
//		end = chrono::high_resolution_clock::now();
//		time = end - start;
//		printf("%2d. Calculation time sequentially:                %0.6f\n", optimisationcounter, time.count());
//		if (PRINTM) printMatrixLong(R, "R sequentially");
//		timetotal[optimisationcounter] += time;
//		optimisationcounter++;
//
//		//Kernels1
//		start = chrono::high_resolution_clock::now();
//		matmult_kernels1(A, B, R);
//		end = chrono::high_resolution_clock::now();
//		time = end - start;
//		printf("%2d. Calculation time naive kernels:               %0.6f\n", optimisationcounter, time.count());
//		if (PRINTM) printMatrixLong(R, "R naive kernels");
//		timetotal[optimisationcounter] += time;
//		optimisationcounter++;
//
//		//Kernels1b
//		start = chrono::high_resolution_clock::now();
//		matmult_kernels1b(A, B, R);
//		end = chrono::high_resolution_clock::now();
//		time = end - start;
//		printf("%2d. Calculation time naive kernels + opt loops:   %0.6f\n", optimisationcounter, time.count());
//		if (PRINTM) printMatrixLong(R, "R naive kernels");
//		timetotal[optimisationcounter] += time;
//		optimisationcounter++;
//
//		//Kernels2
//		start = chrono::high_resolution_clock::now();
//		matmult_kernels2(A, B, R);
//		end = chrono::high_resolution_clock::now();
//		time = end - start;
//		printf("%2d. Calculation time kernels data optimised:      %0.6f\n", optimisationcounter, time.count());
//		if (PRINTM) printMatrixLong(R, "R kernels dataopt");
//		timetotal[optimisationcounter] += time;
//		optimisationcounter++;
//
//		//Kernels3
//		start = chrono::high_resolution_clock::now();
//		matmult_kernels3(A, B, R);
//		end = chrono::high_resolution_clock::now();
//		time = end - start;
//		printf("%2d. Calculation time kernels dataOpt + collapse:  %0.6f\n", optimisationcounter, time.count());
//		if (PRINTM) printMatrixLong(R, "R dataop kernels + colapse");
//		timetotal[optimisationcounter] += time;
//		optimisationcounter++;
//
//		//Parallel1
//		start = chrono::high_resolution_clock::now();
//		matmult_parallel1(A, B, R);
//		end = chrono::high_resolution_clock::now();
//		time = end - start;
//		printf("%2d. Calculation time naive parallel:              %0.6f\n", optimisationcounter, time.count());
//		if (PRINTM) printMatrixLong(R, "R naive parallel");
//		timetotal[optimisationcounter] += time;
//		optimisationcounter++;
//
//		//Parallel2
//		start = chrono::high_resolution_clock::now();
//		matmult_parallel2(A, B, R);
//		end = chrono::high_resolution_clock::now();
//		time = end - start;
//		printf("%2d. Calculation time parallel data optimised:     %0.6f\n", optimisationcounter, time.count());
//		if (PRINTM) printMatrixLong(R, "R dataopt parallel");
//		timetotal[optimisationcounter] += time;
//		optimisationcounter++;
//
//		//Parallel3
//		start = chrono::high_resolution_clock::now();
//		matmult_parallel3(A, B, R);
//		end = chrono::high_resolution_clock::now();
//		time = end - start;
//		printf("%2d. Calculation time parallel dataopt + async:    %0.6f\n", optimisationcounter, time.count());
//		if (PRINTM) printMatrixLong(R, "R dataopt parallel + async");
//		timetotal[optimisationcounter] += time;
//		optimisationcounter++;
//
//		//Parallel4
//		start = chrono::high_resolution_clock::now();
//		matmult_parallel4(A, B, R);
//		end = chrono::high_resolution_clock::now();
//		time = end - start;
//		printf("%2d. Calculation time parallel dataopt + collapse: %0.6f\n", optimisationcounter, time.count());
//		if (PRINTM) printMatrixLong(R, "R dataopt parallel + collapse");
//		timetotal[optimisationcounter] += time;
//		optimisationcounter++;
//
//
//		optimisationcounter = 0;

		//----------------Jacobi ------------------------

		printf("-- Time measuring round %d\n", i);

		//init AJ
		fillRandomMatrixFloat(AJ);
		if (PRINTM) printMatrixFloat(AJ, "AJ");
		printf("Matrix initialised\n");

		//Jacobi Algorithm:
		//Sequential
		copyTmpMatrix(AJ, tmpA);
		start = chrono::high_resolution_clock::now();
		error = jacobi_sequential(tmpA);
		end = chrono::high_resolution_clock::now();
		time = end - start;
		printf("%2d. Calculation time sequentially:                %0.6f, error: %0.9f\n", optimisationcounterj, time.count(), error);
		if (PRINTM) printMatrixFloat(tmpA, "AJ sequentially");
		timetotalj[optimisationcounterj] += time;
		errortotal[optimisationcounterj] += error;
		optimisationcounterj++;

//		//Kernels1
//		copyTmpMatrix(AJ, tmpA);
//		start = chrono::high_resolution_clock::now();
//		error = jacobi_kernels1(tmpA);
//		end = chrono::high_resolution_clock::now();
//		time = end - start;
//		printf("%2d. Calculation time naive kernels:               %0.6f, error: %0.9f\n", optimisationcounterj, time.count(), error);
//		if (PRINTM) printMatrixFloat(tmpA, "AJ naive kernels");
//		timetotalj[optimisationcounterj] += time;
//		errortotal[optimisationcounterj] += error;
//		optimisationcounterj++;
//
//		//Kernels2
//		copyTmpMatrix(AJ, tmpA);
//		start = chrono::high_resolution_clock::now();
//		error = jacobi_kernels2(tmpA);
//		end = chrono::high_resolution_clock::now();
//		time = end - start;
//		printf("%2d. Calculation time kernels data optimised:      %0.6f, error: %0.9f\n", optimisationcounterj, time.count(), error);
//		if (PRINTM) printMatrixFloat(tmpA, "AJ kernels dataopt");
//		timetotalj[optimisationcounterj] += time;
//		errortotal[optimisationcounterj] += error;
//		optimisationcounterj++;
//
//		//Kernels3
//		copyTmpMatrix(AJ, tmpA);
//		start = chrono::high_resolution_clock::now();
//		error = jacobi_kernels3(tmpA);
//		end = chrono::high_resolution_clock::now();
//		time = end - start;
//		printf("%2d. Calculation time kernels dataOpt + collapse:  %0.6f, error: %f\n", optimisationcounterj, time.count(), error);
//		if (PRINTM) printMatrixFloat(tmpA, "AJ dataop kernels + colapse");
//		timetotalj[optimisationcounterj] += time;
//		errortotal[optimisationcounterj] += error;
//		optimisationcounterj++;
//
//		//Parallel1
//		copyTmpMatrix(AJ, tmpA);
//		start = chrono::high_resolution_clock::now();
//		error = jacobi_parallel1(tmpA);
//		end = chrono::high_resolution_clock::now();
//		time = end - start;
//		printf("%2d. Calculation time naive parallel:              %0.6f, error: %0.9f\n", optimisationcounterj, time.count(), error);
//		if (PRINTM) printMatrixFloat(tmpA, "AJ naive parallel");
//		timetotalj[optimisationcounterj] += time;
//		errortotal[optimisationcounterj] += error;
//		optimisationcounterj++;
//
//		//Parallel2
//		copyTmpMatrix(AJ, tmpA);
//		start = chrono::high_resolution_clock::now();
//		error = jacobi_parallel2(tmpA);
//		end = chrono::high_resolution_clock::now();
//		time = end - start;
//		printf("%2d. Calculation time parallel data optimised:     %0.6f, error: %0.9f\n", optimisationcounterj, time.count(), error);
//		if (PRINTM) printMatrixFloat(tmpA, "AJ dataopt parallel");
//		timetotalj[optimisationcounterj] += time;
//		errortotal[optimisationcounterj] += error;
//		optimisationcounterj++;
//
//		//Parallel3
//		copyTmpMatrix(AJ, tmpA);
//		start = chrono::high_resolution_clock::now();
//		error = jacobi_parallel3(tmpA);
//		end = chrono::high_resolution_clock::now();
//		time = end - start;
//		printf("%2d. Calculation time parallel dataopt + async:    %0.6f, error: %0.9f\n", optimisationcounterj, time.count(), error);
//		if (PRINTM) printMatrixFloat(tmpA, "AJ dataopt parallel + async");
//		timetotalj[optimisationcounterj] += time;
//		errortotal[optimisationcounterj] += error;
//		optimisationcounterj++;

		//Parallel4
		copyTmpMatrix(AJ, tmpA);
		start = chrono::high_resolution_clock::now();
		error = jacobi_parallel4(tmpA);
		end = chrono::high_resolution_clock::now();
		time = end - start;
		printf("%2d. Calculation time parallel dataopt + collapse: %0.6f, error: %0.9f\n", optimisationcounterj, time.count(), error);
		if (PRINTM) printMatrixFloat(tmpA, "AJ dataopt parallel + collapse");
		timetotalj[optimisationcounterj] += time;
		errortotal[optimisationcounterj] += error;
		optimisationcounterj++;

		//Parallel5
		copyTmpMatrix(AJ, tmpA);
		start = chrono::high_resolution_clock::now();
		error = jacobi_parallel5(tmpA);
		end = chrono::high_resolution_clock::now();
		time = end - start;
		printf("%2d. Calculation time parallel dataopt + collapse: %0.6f, error: %0.9f\n", optimisationcounterj, time.count(), error);
		if (PRINTM) printMatrixFloat(tmpA, "AJ dataopt parallel + collapse");
		timetotalj[optimisationcounterj] += time;
		errortotal[optimisationcounterj] += error;
		optimisationcounterj++;

		optimisationcounterj = 0;

	}

	for (int i = 0; i < noOptsJ; i++) {
		timetotalj[i] = timetotalj[i]/LOOPS;
		errortotal[i] = errortotal[i]/LOOPS;
		timetotal[i]  = timetotal[i]/LOOPS;
	}
		timetotal[noOpts-1] = timetotal[noOpts-1]/LOOPS; //matmult hat eine stelle mehr

	printf("########################################################################################################\n");
	printf("Matrix with %d x %d elements\n", DIMXJ, DIMYJ);
	printf("Time average of %d iterations:\n\n", LOOPS);
	printf("Sequential:       %0.6fs, error: %0.6f\n\n",
	       timetotalj[0], errortotal[0]);
//	printf("Naive Kernels:    %0.6fs, error: %0.6f\nDataOpt Kernels:  %0.6fs, error: %0.6f\nCollapse Kernels: %0.6fs, error: %0.6f\n\n",
//	       timetotalj[1], errortotal[1], timetotalj[2], errortotal[2], timetotalj[3], errortotal[3]);
//	printf("Naive Parallel:   %0.6fs, error: %0.6f\nDataOpt Parallel: %0.6fs, error: %0.6f\nAsync Parallel:   %0.6fs, error: %0.6f\nCollapse Parallel:%0.6fs, error: %0.6f\n",
//	       timetotalj[4], errortotal[4], timetotalj[5], errortotal[5], timetotalj[6], errortotal[6], timetotalj[7], errortotal[7]);
	printf("Collapse Parallel:%0.6fs, error: %0.6f\nCollapse Parallel:%0.6fs, error: %0.6f\n",
	       timetotalj[1], errortotal[1], timetotalj[2], errortotal[2]);

//	printf("########################################################################################################\n");
//	printf("########################################################################################################\n");
//
//	printf("Matrices with %d x %d and %d x %d elements\n", DIMX, DIMY, DIMZ, DIMX);
//	printf("Time average of %d iterations:\n\n", LOOPS);
//	printf("Sequential:           %0.6fs\n\n",
//	       timetotal[0]);
//	printf("Naive Kernels:        %0.6fs\nKernels + opt Loobody:%0.6fs\nDataOpt Kernels:      %0.6fs\nCollapse Kernels:     %0.6fs\n\n",
//	       timetotal[1], timetotal[2], timetotal[3], timetotal[4]);
//	printf("Naive Parallel:       %0.6fs\nDataOpt Parallel:     %0.6fs\nAsync Parallel:       %0.6fs\nCollapse Parallel:    %0.6fs\n",
//	       timetotal[5], timetotal[6], timetotal[7], timetotal[8]);
}
