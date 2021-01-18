//
// Created by eva on 30.11.20.
//

#include <cstdio>

/**
 * Multiplies two matrices A and B seuqentially
 */
void mult_seq(long *result, long *a, long *b, int height, int width, int widthA) {
for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			// walk through all cells of a row in matrixA and every cell in a column of matrixB
			// and multiply
			for (int i = 0; i < widthA; i++) {
				result[row * width + col] += a[row * widthA + i] * b[i * width + col];
			}
		}
	}
}

/**
 * Multiplies two matrices A and B seuqentially
 * But with switched inner loops
 */
void mult_seq_speed(long *result, long *a, long *b, int height, int width, int widthA) {
	for (int row = 0; row < height; row++) {
		for (int i = 0; i < widthA; i++) {
			for (int col = 0; col < width; col++) {
				result[row * width + col] += a[row * widthA + i] * b[i * width + col];
			}
		}
	}
}

/**
 * Multiplies two matrices A and B sequentially
 * But with switched inner loops
 */
void mult_seq_speed_cache(long *result, long *a, long *b, int height, int width, int widthA) {
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			long sum = 0;
			for (int i = 0; i < widthA; i++) {
				sum += a[row * widthA + i] * b[i * width + col];
			}
			result[row * width + col] = sum;
		}
	}
}


