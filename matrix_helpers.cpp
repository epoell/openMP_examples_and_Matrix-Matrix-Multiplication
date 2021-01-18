//
// Created by eva on 30.11.20.
//

#include "matrix_helpers.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <omp.h>

using namespace std;

/**
 * Fill a Matrix with random numbers (and sparsity)
 */
void fillRandomMatrix(long *matrix, int height, int width, int range) {
    for (int r = 0; r < height; r++)
        for (int c = 0; c < width; c++) {
            matrix[r * width + c] = rand() % range;
        }
}

/**
 * Reset a Matrix with zeros
 */
void resetMatrix(long *matrix, int height, int width) {
    for (int r = 0; r < height; r++)
        for (int c = 0; c < width; c++) {
            matrix[r * width + c] = 0;
        }
}

/**
 * Print given Matrix
 */
void printMatrix(long *matrix, int height, int width, string name) {
    cout << "Matrix: " << name << "\n";

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            cout << matrix[r * width + c] << "\t";
        }
        cout << "\n";
    }
    cout << "\n";
}

/**
 * Calculates the average value of a given long-array
 * @param array
 * @param length
 * @return the average as double
 */
float calcAvg(float *array, int length) {
    float sum = 0.0;
    for (int i=0; i<length; i++) {
        sum += array[i];
    }
    return (float) sum/length;
}

/**
 * Prety print elapsed time of an algorithm
 * @param duration the measured time
 * @param algorithm  the name for he algorithm
 */
void printTimeDiff(long duration, string algorithm) {
    cout << "Duration for "<< algorithm << ": \t" << duration << " microseconds\n";
}

long checkCorrectness(long *matrix, long *reference, int height, int width) {
	int row, col;
	long diff = 0;
	long diff_i = 0;
	for (row=0; row<height; row++) {
		for (col = 0; col < width; col++) {
			if (matrix[row * width + col] != reference[row * width + col]) {
				diff_i = abs(matrix[row * width + col] - reference[row * width + col]);
//				printf("Difference at [%i][%i]: %ld\n", row, col, diff_i);
				diff += diff_i;
			}
		}
	}
	return diff;
}

void printAverages(string *alg_names, float *alg_averages, int limit, int linepoint) {
    //print averages
    printf("Compare run times:\n");
    printf("------------------\n");
    for (int a = 0; a < limit; a++) {
        printf("%s\t |%.0f| mls\n", alg_names[a].c_str(), alg_averages[a]);
        if ((a%linepoint) == 1) printf("------------------\n");
    }
}

/**
 * Prints information and measures the time for a given algorithm
 *
 * @param func The matrix multiplication algorithm to be measured
 * @param result The result matrix
 * @param a Input matrix 1
 * @param b Input matrix 2
 * @param heightR y-Dimension of result matrix
 * @param widthR x-Dimension of result matrix
 * @param widthA  x-Dimension of Input matrix 1
 * @param duration an array to store teh time used on the algorithm
 * @param loops Number of iterations the multiplication takes place
 * @param print If the result matrix should be printed or not
 * @param description The name of the algorithm
 *
 * @return the average of durations
 */
float doTheMeasuring(void func(long *result, long *a, long *b, int heightR, int widthR, int widthA),
                    long *result, long *a, long *b, int heightR, int widthR, int widthA,
                    int loops, bool print, long *refMatrix, string description){
    float duration[loops];
    long error;
    bool errorBool = false;

    printf("Alorithm: %s\n", description.c_str());
    printf("----------------------------------\n");
    for (int loop = 0; loop < loops; loop++) {
        resetMatrix(result, heightR, widthR);
        duration[loop] = 0;
        auto begin = chrono::high_resolution_clock::now();
        func(result, a, b, heightR, widthR, widthA);
        auto end = chrono::high_resolution_clock::now();
        error = checkCorrectness(result, refMatrix, heightR, widthR);
        if (error > 0) errorBool = true;
        duration[loop] = (float) chrono::duration_cast<chrono::microseconds>(end - begin).count() / 1000;
        if (print) printMatrix(result, heightR, widthR, description);
        printf("-- %i. Loop Duration:       \t %.3f mls\n", loop, duration[loop]);
        printf("      Error in calculation:\t %ld\n", error);
    }
    if (errorBool){
    	printf("Errors\n\n");
    	return -1;
    }
    float avg = calcAvg(duration, loops);
    printf("Average Duration:\t %.0f mls\n\n", avg);
    return avg;
}

float doIncreasingMeasuring(void func(long *result, long *a, long *b, int heightR, int widthR, int widthA, int threads),
                            long *result, long *a, long *b, int size,
                            int loops, long *refMatrix, string description, int threads){
    float duration[loops];
    long error;
    bool errorBool = false;

    printf("Alorithm: %s\n", description.c_str());
    printf("----------------------------------\n");
    for (int loop = 0; loop < loops; loop++) {
        resetMatrix(result, size, size);
        duration[loop] = 0;
        auto begin = chrono::high_resolution_clock::now();
        func(result, a, b, size, size, size, threads);
        auto end = chrono::high_resolution_clock::now();
        error = checkCorrectness(result, refMatrix, size, size);
        if (error > 0) errorBool = true;
        duration[loop] = (float) chrono::duration_cast<chrono::microseconds>(end - begin).count();
        printf("-- %i. Loop Duration:       \t %.3f mys\n", loop, duration[loop]);
        printf("      Error in calculation:\t %ld\n", error);
    }
    if (errorBool){
        printf("Errors\n\n");
        return -1;
    }
    float avg = calcAvg(duration, loops);
    printf("Average Duration:\t %.0f mys\n\n", avg);
    return avg;

}