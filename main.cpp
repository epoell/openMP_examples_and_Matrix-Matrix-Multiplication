#include <cstdio>
#include "min_examples.h"
#include "matmult_measuring.h"

using namespace std;


int main() {
    //print to file instead of console
    bool print_to_file = false;
    FILE *file;
    if (print_to_file) {

        if ((file = freopen("Parallel-Prgming.txt", "w", stdout)) == NULL) {
            printf("Cannot open file\n");
        }
    }

	//Minimal Examples of OpenMP directives
	//scheduling example takes very long, so only run it once
	bool schedule = true;
//	for (int i=0; i<3; i++) {
		minimal_examples(schedule);
		schedule = false;
//	}

    //Matrix Multiplication
    //---------------------
    //Make Matrix x Matrix Multiplication faster with different directives
//    for (int i = 0; i < 5; i++) {
        matmult_buildup();
//    }

    //Compare basic and fastes directive along cores
//    for (int i = 0; i < 2; i++) {
        matmult_increase_threads();
//    }

    //Compare sequential and fastes directive along Matrix size
//    for (int i = 0; i < 2; i++) {
        matmult_increase_size();
//    }

    if (print_to_file) {
        fclose(file);
    }

    return 0;
}



