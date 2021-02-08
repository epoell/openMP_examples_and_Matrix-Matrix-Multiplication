# OpenMP Examples
This repository hold the programming code of a study project on parallel programming on CPUs with OpenMP. To illustrate my text, I tried to give minimal examples on common OpenMP pragmas and accelerate the execution of a matrix-matrix-multiplication. Because I did not find an extensive repository about this, I wanted to share my findings here.

## Table of Contents
[Structure of the repository](#structure)  
[Minimal Examples](#expls)  
[Matrix-Matrix-Multiplication](#mult)  
[How to execute it](#exec)  


## Structure of the repository
<a name="structure"/>
The repository is structured as follows:

- **main.cpp** combines all experiments
- **min_examples.cpp/.h** holds the code for the minimal examples on common pragmas
- **matmult_functions.h** combines the signatures of all functions to succeedingly accelerate execution of matrix-matrix-multiplication
- **matmult_sequential.cpp** multiplies two matrices sequentially with increasingly memory efficient algorithms
- **matmult_parallel.cpp** uses different OpenMP pragmas to multiply the matrices in parallel
- **matmult_measuring.cpp/.h** is called by the main-function and itself calls the multiplication functions. It takes care of time measuring and a structured output of the experiments
- **matrix_helpers.cpp/.h** is a collection of helper functions to measure the time of the multiplication, initialise or reset matrices and check the correctness of the matrix product

## Minimal Examples
<a name="expls"/>
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

## Matrix-Matrix-Multiplication
<a name="mult"/>
The actual multiplication algorithms are in `matmult_sequential.cpp` and `matmult_parallel.cpp`. The matrices are built as one long array rather than using a two dimeansional array as the size can be defined more variably in C++. The matrix-data is concatenated rowwise. Thus access of each cell needs to be calculated: instead of `[row][column]` it is given as `[(row * martix-width) + column]`.

`matmult_sequential.cpp`:

- `mult_seq`: describes the naive version of muliplying two matrices with three loops going thourgh the rows of input matrix A, the columns of input matrix B and the cells within these.
- `mult_seq_speed`: switching the order of the loops is more memory effective, and speeds up the execution already. This algorithm is still sequenial.
- `mult_seq_speed_cache`: uses the naive loop order but stores the procuct of the cells in an intermediate variable. This uses the cache more efficient and again results in a speedup compared to both other sequential varaints.


In `matmult_parallel.cpp` OpenMP is used to accelerate the multiplication with different pragmas. The Speed ups are compared to `mult_seq_speed_cache` and run on 4 cores.

- `mult_basic`: uses the algorithm of `mult_seq_speed` with `parallel for`. Speed up: 3.12
`mult_basic_threads_size`: is used to compare the execution time of he basic algorithm along increasingly many threads
- `mult_3for`: does not execute correctly; it uses  `parallel for` for each loop
- `mult_basic_private`:  uses the algorithm of `mult_seq_speed` with `parallel for private(row, i, col)`. Speed up: 2.46
- `mult_basic_private_shared`: uses the algorithm of `mult_seq_speed` with `parallel for private(row, i, col) shared(a, b, result)`. Speed up: 3.13
- `mult_basic_private_private`: uses the algorithm of `mult_seq_speed` with `private(row, i, col) firstprivate (a, b) shared (result)`. Speed up: 3.11
- `mult_basic_private_switchedloops`:  uses the algorithm of `mult_seq` with `parallel for private(row, i, col)`. Speed up: 2.46
- `mult_collapse3`: uses the algorithm of `mult_seq_speed` with `parallel for collapse(3)`. Speed up: 2.88
- `mult_collapse2`: uses the algorithm of `mult_seq_speed` with `parallel for collapse(2)`. Speed up: 3.12
- `mult_collapse2_private`: uses the algorithm of `mult_seq_speed` with `parallel for collapse(2) private(row, col, i)`. Speed up: 3.10
- `mult_collapse2_private_cache`: uses the algorithm of `mult_seq_speed_cache` with `parallel for collapse(2) private(row, col, i)`. Speed up: 3.67
`mult_collapse2_private_cache_threads_size`: is used to compare the execution time of he basic algorithm along increasingly many threads


## How to execute it
<a name="exec"/>
