#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MATRIX_DIM 20
#define BLOCK_DIM 5
#define NUM_PROCS_EXPECTED 4 // it should run on four processes

// Dimensions for local data on each process
#define LOCAL_MATRIX_ROWS MATRIX_DIM
#define LOCAL_MATRIX_COLS (MATRIX_DIM / NUM_PROCS_EXPECTED) // 20/4=5
#define LOCAL_VECTOR_X_SIZE (MATRIX_DIM / NUM_PROCS_EXPECTED) // 20/4=5
#define GLOBAL_RESULT_VECTOR_SIZE MATRIX_DIM

const char* MATRIX_FILENAME = "mat-d20-b5-p4.bin";
const char* VECTOR_X_FILENAME = "x-d20.txt.bin";

// Function to print a matrix (row-major storage)
void print_matrix(const char* title, double* matrix, int rows, int cols, int rank) {
    printf("---- Process %d: %s (%d x %d) ----\n", rank, title, rows, cols);
    for (int i = 0; i < rows; ++i) {
        printf("P%d: ", rank);
        for (int j = 0; j < cols; ++j) {
            printf("%8.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

// Function to print a vector
void print_vector(const char* title, double* vector, int size, int rank) {
    printf("---- Process %d: %s (%d x 1) ----\n", rank, title, size);
    printf("P%d: ", rank);
    for (int i = 0; i < size; ++i) {
        printf("%8.2f ", vector[i]);
    }
    printf("\n");
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (num_procs != NUM_PROCS_EXPECTED) {
        if (rank == 0) {
            fprintf(stderr, "Error: This program only runs on %d processes.\n", NUM_PROCS_EXPECTED);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if (LOCAL_MATRIX_COLS != BLOCK_DIM || LOCAL_VECTOR_X_SIZE != BLOCK_DIM) {
         if (rank == 0) {
            fprintf(stderr, "Error: Mismatch in calculated local dimensions and block dimension.\n");
            fprintf(stderr, "LOCAL_MATRIX_COLS = %d, LOCAL_VECTOR_X_SIZE = %d, BLOCK_DIM = %d\n",
                    LOCAL_MATRIX_COLS, LOCAL_VECTOR_X_SIZE, BLOCK_DIM);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }


    // Allocate memory for local matrix, local vector x, and local result contribution
    double* local_A = (double*)malloc(LOCAL_MATRIX_ROWS * LOCAL_MATRIX_COLS * sizeof(double));
    double* local_x = (double*)malloc(LOCAL_VECTOR_X_SIZE * sizeof(double));
    double* local_y_contribution = (double*)calloc(GLOBAL_RESULT_VECTOR_SIZE, sizeof(double)); // Initialize to 0
    double* global_y = NULL;

    if (!local_A || !local_x || !local_y_contribution) {
        fprintf(stderr, "P%d: Failed to allocate local memory.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        global_y = (double*)malloc(GLOBAL_RESULT_VECTOR_SIZE * sizeof(double));
        if (!global_y) {
             fprintf(stderr, "P%d: Failed to allocate global_y memory.\n", rank);
             MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }


    MPI_File fh_matrix, fh_vector_x;
    MPI_Status status;
    int file_dim;

    // Read the matrix
    MPI_File_open(MPI_COMM_WORLD, MATRIX_FILENAME, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_matrix);
    if (fh_matrix == MPI_FILE_NULL) {
        if (rank == 0) fprintf(stderr, "Error opening matrix file: %s\n", MATRIX_FILENAME);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Rank 0 reads and broadcasts the dimension (or all can read, simpler for one int)
    if (rank == 0) {
        MPI_File_read(fh_matrix, &file_dim, 1, MPI_INT, &status);
    }
    MPI_Bcast(&file_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);


    MPI_Offset matrix_offset = sizeof(int) + (MPI_Offset)rank * LOCAL_MATRIX_ROWS * LOCAL_MATRIX_COLS * sizeof(double);
    MPI_File_read_at_all(fh_matrix, matrix_offset, local_A, LOCAL_MATRIX_ROWS * LOCAL_MATRIX_COLS, MPI_DOUBLE, &status);
    MPI_File_close(&fh_matrix);

    // Read vector
    MPI_File_open(MPI_COMM_WORLD, VECTOR_X_FILENAME, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_vector_x);
     if (fh_vector_x == MPI_FILE_NULL) {
        if (rank == 0) fprintf(stderr, "Error opening vector file: %s\n", VECTOR_X_FILENAME);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        MPI_File_read(fh_vector_x, &file_dim, 1, MPI_INT, &status);

    }
    MPI_Bcast(&file_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);


    MPI_Offset vector_x_offset = sizeof(int) + (MPI_Offset)rank * LOCAL_VECTOR_X_SIZE * sizeof(double);
    MPI_File_read_at_all(fh_vector_x, vector_x_offset, local_x, LOCAL_VECTOR_X_SIZE, MPI_DOUBLE, &status);
    MPI_File_close(&fh_vector_x);

    // Print local data; note that we synchronize implying we might have some timing issues here but timing is not the purpose of this assignment
    for (int i = 0; i < num_procs; ++i) {
        if (rank == i) {
            print_matrix("Local Matrix A_p", local_A, LOCAL_MATRIX_ROWS, LOCAL_MATRIX_COLS, rank);
            print_vector("Local Vector x_p", local_x, LOCAL_VECTOR_X_SIZE, rank);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    //  Local Matrix-Vector Multiplication
    // local_A is (LOCAL_MATRIX_ROWS x LOCAL_MATRIX_COLS)
    // local_x is (LOCAL_VECTOR_X_SIZE x 1)
    for (int i = 0; i < LOCAL_MATRIX_ROWS; ++i) {
        local_y_contribution[i] = 0.0;
        for (int j = 0; j < LOCAL_MATRIX_COLS; ++j) {
            local_y_contribution[i] += local_A[i * LOCAL_MATRIX_COLS + j] * local_x[j];
        }
    }

    // Reduce all local_y_contribution vectors to get the global result y; sum local_y_contribution to get the final solution
    MPI_Reduce(local_y_contribution, global_y, GLOBAL_RESULT_VECTOR_SIZE, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Print final result
    if (rank == 0) {
        print_vector("Final Result Vector y = Ax", global_y, GLOBAL_RESULT_VECTOR_SIZE, rank);
    }

    // Free
    free(local_A);
    free(local_x);
    free(local_y_contribution);
    if (rank == 0) {
        free(global_y);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
