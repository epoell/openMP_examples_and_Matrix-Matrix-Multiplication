# OpenMP Examples
This repository hold the programming code of a study project on parallel programming on CPUs with OpenMP. To illustrate my text, I tried to give minimal examples on common OpenMP pragmas and accelerate the execution of a matrix-matrix-multiplication.

## Structure of the repository
The repository is structured as follows:

- **main.cpp** combines all experiments
- **min_examples** holds the code for the minimal examples on common pragmas
- **matmult_functions.h** combines the signatures of all functions to succeedingly accelerate execution of matrix-matrix-multiplication
- **matmult_sequential.cpp** multiplies two matrices sequentially with increasingly memory efficient algorithms
- **matmult_parallel.cpp** uses different OpenMP pragmas to multiply the matrices in parallel
- **matmult_measuring** is called by the main-function and itself calls the multiplication functions. It takes care of time measuring and a structured output of the experiments
- **matrix_helpers** is a collection of helper functions to measure the time of the multiplication, initialise or reset matrices and check the correctness of the matrix product

## Minimal Examples
The minimal examples include the following pragmas, options and functions:

- parallel
- num_threads(x)
- single
- master
- nowait
- barrier
- for
- private(x) firstprivate(y) shared(s)
- collapse(x)
- reduction(max:m)
- schedule(static), schedule(dynamic), schedule(guided), schedule(auto)
- sections, section
- task, taskwait
- atomic
- omp_set_nested(x)
- omp_get_num_procs()
- omp_get_max_threads()
- omp_get_thread_num()
- omp_get_num_teams()
- omp_get_team_num()
- omp_get_team_size(omp_get_level())
- omp_set_num_threads(x)

## Matrix Multiplication
## How to execute it
