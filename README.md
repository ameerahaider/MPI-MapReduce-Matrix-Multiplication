# -MPI-MapReduce-Matrix-Multiplication
Explore the power of parallel computing with MPI in C through the MPI MapReduce Matrix Multiplication project. This repository showcases an implementation of the MapReduce framework using MPI, simulating Hadoop's architecture for task distribution across nodes in a cluster.  

# MPI MapReduce Matrix Multiplication

## Project Description
This project demonstrates the implementation of the MapReduce framework using MPI in C to perform matrix multiplication. It emulates the Hadoop architecture for distributing tasks among nodes in a cluster. The project focuses on utilizing parallel and distributed computing techniques to achieve efficient matrix multiplication.

## Project Structure
- `main.c`: The main C program that performs MapReduce matrix multiplication using MPI.
- `input_file_A.txt` and `input_file_B.txt`: Input files containing matrices A and B, respectively.
- `output_file.txt`: Output file containing the result of matrix multiplication.

## Requirements
- MPI library and environment setup.
- Beowulf Cluster for distributed execution.
- C compiler (e.g., GCC).

## How to Run

1. **Compile the Code:**
    ```bash
    mpicc -o matrix_multiplication main.c -lm
    ```

2. **Generate Input Matrices:**
    Before running the program, you need to create input matrices. You can either use provided input files or generate new matrices using the following command:
    ```bash
    ./matrix_multiplication <matrix_size> input_A.txt input_B.txt
    ```
    Replace `<matrix_size>` with the desired size of square matrices.

3. **Run on Beowulf Cluster:**
    - Transfer the compiled executable and input files to the cluster.
    - Launch the program using `mpiexec` with the desired number of processes:
    ```bash
    mpiexec -np <num_processes> ./matrix_multiplication <matrix_size> input_A.txt input_B.txt output_matrix.txt
    ```
    Replace `<num_processes>` with the total number of processes you want to use.

4. **Check Output:**
    After the program completes, the result of matrix multiplication will be saved in `output_file.txt`.

5. **Validation:**
    Compare the generated `output_file.txt` with the result obtained from a serial matrix multiplication program to ensure correctness.

## Output
- The program will print task assignments and completion status for each process.
- The master process will compare the MPI-generated output with a serial matrix multiplication output.

## Project Notes
- The project aims to demonstrate MapReduce using MPI, with a focus on emulating the Hadoop architecture.
- The code may be executed on a Beowulf Cluster for distributed computation.
