//
// Created by eva on 30.11.20.
//

#pragma once

#include <string>
#include <chrono>

/**
 * Fill a Matrix with random numbers (and sparsity)
 */
void fillRandomMatrix(long *matrix, int height, int width, int range);

/**
 * Reset a Matrix with zeros
 */
void resetMatrix(long *matrix, int height, int width);

/**
 * Print given Matrix
 */
void printMatrix(long *matrix, int height, int width, std::string name);

/**
 * Calculates the average value of a given long-array
 * @param array
 * @param length
 * @return the average as double
 */
float calcAvg(long *array, int length);

/**
 * Prety print elapsed time of an algorithm
 * @param duration the measured time
 * @param algorithm  the name for he algorithm
 */
void printTimeDiff(std::chrono::duration<long int, std::ratio<1, 1000000> > duration, std::string algorithm);

long checkCorrectness(long *matrix, long *reference, int height, int width);

void printAverages(std::string *alg_names, float *alg_averages, int limit, int linepoint);

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
                    int loops, bool print, long *refMatrix, std::string description);
float doIncreasingMeasuring(void func(long *result, long *a, long *b, int heightR, int widthR, int widthA, int threads),
                            long *result, long *a, long *b, int size,
                            int loops, long *refMatrix, std::string description, int threads);