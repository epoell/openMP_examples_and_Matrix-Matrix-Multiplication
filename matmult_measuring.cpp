//
// Created by eva on 14.01.21.
//

#include "matmult_measuring.h"
#include <omp.h>
#include <cstdio>
#include "matmult_functions.h"
#include "matrix_helpers.h"

using namespace std;

//Global Constants
// result matrix: heightA (rows) x widthB (cols)
// widthA == heightB, not in result matrix.
const int size = 500;
const int HEIGHT_A = size;
const int WIDTH_A = size;
const int HEIGHT_B = WIDTH_A;
const int WIDTH_B = size;
const int HEIGHT_R = HEIGHT_A;
const int WIDTH_R = WIDTH_B;

const int loops = 3;

/**
 * Test different Omp directives with Matrix Multiplication
 */
void matmult_buildup() {
	printf(     "\n\n-------------------------\n"
	            "| Matrix Multiplication |\n"
	            "-------------------------\n\n");
	//view matrices sequentially instead of 2d is already memory friendly!!
	int algorithms = 30;
	string alg_names[algorithms];
	float alg_averages[algorithms];

	bool print = false;

	long A[HEIGHT_A * WIDTH_A] = {};
	long B[HEIGHT_B * WIDTH_B] = {};
	long Result[HEIGHT_R * WIDTH_R] = {};
	long Ref[HEIGHT_R * WIDTH_R];

	fillRandomMatrix(A, HEIGHT_A, WIDTH_A, 10);
	fillRandomMatrix(B, HEIGHT_B, WIDTH_B, 10);
	if (print) printMatrix(A, HEIGHT_A, WIDTH_A, "A");
	if (print) printMatrix(B, HEIGHT_B, WIDTH_B, "B");

	omp_set_num_threads(4);
	int i = 0;
	alg_names[i] = "Sequenial MatMult                                   ";
	alg_averages[i] = doTheMeasuring(mult_seq,
	                                 Ref, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
	i++;
//	alg_names[i] = "Seq with inversed loops                             ";
//	alg_averages[i] = doTheMeasuring(mult_seq_speed,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
//	alg_names[i] = "Seq with inversed loops and caching                 ";
//	alg_averages[i] = doTheMeasuring(mult_seq_speed_cache,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
//
	alg_names[i] = "Parallel MatMult basic                              ";
	alg_averages[i] = doTheMeasuring(mult_basic,
	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
	i++;
//	alg_names[i] = "Parallel MatMult several for directives             ";
//	alg_averages[i] = doTheMeasuring(mult_3for,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
//	alg_names[i] = "Parallel MatMult basic with private vars            ";
//	alg_averages[i] = doTheMeasuring(mult_basic_private_switchedloops,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
//	alg_names[i] = "Parallel MatMult basic with priv, shared vars       ";
//	alg_averages[i] = doTheMeasuring(mult_basic_private_shared,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
	alg_names[i] = "Parallel MatMult basic with priv, priv matrices     ";
	alg_averages[i] = doTheMeasuring(mult_basic_private_shared,
	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
	i++;
//	alg_names[i] = "Parallel MatMult all 3 loops collapsed              ";
//	alg_averages[i] = doTheMeasuring(mult_collapse3,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
//	alg_names[i] = "Parallel MatMult 2 outer loops collapsed            ";
//	alg_averages[i] = doTheMeasuring(mult_collapse2,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
//	alg_names[i] = "Parallel MatMult 2 outer loops collapsed + priv     ";
//	alg_averages[i] = doTheMeasuring(mult_collapse2_private,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
//	//2nd Fastest!
//	alg_names[i] = "Parallel MatMult 2 loops collapsed + priv + cache   ";
//	alg_averages[i] = doTheMeasuring(mult_collapse2_private_cache,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
//	//Fastest!
//	alg_names[i] = "Parallel MatMult reduction for outer loop           ";
//	alg_averages[i] = doTheMeasuring(mult_reduction,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
//	alg_names[i] = "Parallel MatMult reduction in innerest loop         ";
//	alg_averages[i] = doTheMeasuring(mult_reduction_inner,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
//	alg_names[i] = "Parallel MatMult reduction in innerest + collapse   ";
//	alg_averages[i] = doTheMeasuring(mult_reduction_inner_collapse,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
//	alg_names[i] = "Parallel MatMult static scheduling                  ";
//	alg_averages[i] = doTheMeasuring(mult_schedule_static,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
//	alg_names[i] = "Parallel MatMult static scheduling + chunk size 125 ";
//	alg_averages[i] = doTheMeasuring(mult_schedule_static_chunk,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
//	alg_names[i] = "Parallel MatMult static scheduling + dynm chunk size";
//	alg_averages[i] = doTheMeasuring(mult_schedule_static_chunk_dynamic,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
//	alg_names[i] = "Parallel MatMult dynamic scheduling                 ";
//	alg_averages[i] = doTheMeasuring(mult_schedule_dynamic,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
//	alg_names[i] = "Parallel MatMult dynamic scheduling + chunk size 4  ";
//	alg_averages[i] = doTheMeasuring(mult_schedule_dynamic_chunk,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
//	alg_names[i] = "Parallel MatMult guided scheduling                  ";
//	alg_averages[i] = doTheMeasuring(mult_schedule_guided,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
//	alg_names[i] = "Parallel MatMult guided scheduling + chunk size 125 ";
//	alg_averages[i] = doTheMeasuring(mult_schedule_guided_chunk,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;
//	alg_names[i] = "Parallel MatMult automatic scheduling               ";
//	alg_averages[i] = doTheMeasuring(mult_schedule_auto,
//	                                 Result, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, print, Ref, alg_names[i]);
//	i++;

	//print averages
	printf("Compare run times:\n");
	printf("------------------\n");
	for (int a = 0; a < i; a++) {
		printf("%s\t |%.0f| mls\n", alg_names[a].c_str(), alg_averages[a]);
		if (a == 2) printf("------------------\n");
	};
}

void matmult_increase_threads() {
	int size = 500;

	long A[size * size];
	long B[size * size];
	long Result[size * size];
	long Ref[size * size];

	fillRandomMatrix(A, size, size, 10);
	fillRandomMatrix(B, size, size, 10);

    string alg_names[50];
    float alg_averages[50];
    int loops = 5;
    int i = 0;

    //Sequential
    // basic
	alg_names[i] = "Sequenial MatMult                                 1 Thread";
	alg_averages[i] = doTheMeasuring(mult_seq,
	                                 Ref, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, false, Ref, alg_names[i]);
	i++;
    // cached
    alg_names[i] = "Seq with inversed loops                           1 Thread";
	alg_averages[i] = doTheMeasuring(mult_seq_speed,
	                                 Ref, A, B, HEIGHT_R, WIDTH_R, WIDTH_A, loops, false, Ref, alg_names[i]);
	i++;

    //Parallel
    for(int threads=1; threads<=12; threads++) {
		// basic
		alg_names[i] = "Parallel MatMult basic                            "+to_string(threads)+" Thread(s)";
		alg_averages[i] = doIncreasingMeasuring(mult_basic_threads_size,
		                                        Result, A, B, size, loops, Ref, alg_names[i], threads);
		i++;
		// cached
		alg_names[i] = "Parallel MatMult 2 loops collapsed + priv + cache "+to_string(threads)+" Thread(s)";
		alg_averages[i] = doIncreasingMeasuring(mult_collapse2_private_cache_threads_size,
		                                        Result, A, B, size, loops, Ref, alg_names[i], threads);
		i++;
		// reduces
		alg_names[i] = "Parallel MatMult reduction for outer loop         "+to_string(threads)+" Thread(s)";
		alg_averages[i] = doIncreasingMeasuring(mult_reduction_threads_size,
		                                        Result, A, B, size, loops, Ref, alg_names[i], threads);
		i++;
	}

    printAverages(alg_names, alg_averages, i, -1);
}

void matmult_increase_size() {
	int loops = 5;
	int x = 44 * loops * 5; //iterations * loops * algorithms
	string alg_names[x];
	float alg_averages[x];
	int i = 0;
	omp_set_num_threads(4);

	for (int size=4; size<=508; size+=12) { //44 iterations
		printf("\nSize %i\n--------\n", size);

		long A[size * size];
		long B[size * size];
		long Result[size * size];
		long Ref[size * size];

		fillRandomMatrix(A, size, size, 10);
		fillRandomMatrix(B, size, size, 10);

		//Sequential
		// basic
		alg_names[i] = "Sequenial MatMult                                 ";
		alg_averages[i] = doTheMeasuring(mult_seq,
		                                 Ref, A, B, size, size, size, loops, false, Ref, alg_names[i]);
		i++;
		// cached
		alg_names[i] = "Seq with inversed loops                           ";
		alg_averages[i] = doTheMeasuring(mult_seq_speed,
		                                 Ref, A, B, size, size, size, loops, false, Ref, alg_names[i]);
		i++;

		//Parallel
		// basic
		alg_names[i] = "Parallel MatMult basic                            ";
		alg_averages[i] = doIncreasingMeasuring(mult_basic_threads_size,
		                                        Result, A, B, size, loops, Ref, alg_names[i], 4);
		i++;
		// cached
		alg_names[i] = "Parallel MatMult 2 loops collapsed + priv + cache ";
		alg_averages[i] = doIncreasingMeasuring(mult_collapse2_private_cache_threads_size,
		                                        Result, A, B, size, loops, Ref, alg_names[i], 4);
		i++;
		// reduces
		alg_names[i] = "Parallel MatMult reduction for outer loop         ";
		alg_averages[i] = doIncreasingMeasuring(mult_reduction_threads_size,
		                                        Result, A, B, size, loops, Ref, alg_names[i], 4);
		i++;
	}

	printAverages(alg_names, alg_averages, i, 5);
}
