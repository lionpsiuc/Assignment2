#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MATRIX_FILE "mat-d20-b5-p4.bin"
#define VECTOR_FILE "x-d20.txt.bin"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if the number of processes is 4
    if (size != 4) {
        if (rank == 0) printf("Need 4 processes\n");
        MPI_Finalize();
        return 1;
    }

    // Open and read the matrix file
    MPI_File mat_file;
    MPI_File_open(MPI_COMM_WORLD, MATRIX_FILE, MPI_MODE_RDONLY, MPI_INFO_NULL, &mat_file);

    int matrix_dim;
    MPI_File_read(mat_file, &matrix_dim, 1, MPI_INT, MPI_STATUS_IGNORE);
    // Check if the matrix dimension is 20
    if (matrix_dim != 20) {
        printf("Matrix dimension error\n");
        MPI_File_close(&mat_file);
        MPI_Finalize();
        return 1;
    }

    const int block_size = 5;
    double* local_matrix = (double*)malloc(block_size * matrix_dim * sizeof(double));
    // Check if memory allocation for the local matrix is successful
    if (local_matrix == NULL) {
        printf("Memory error for matrix\n");
        MPI_File_close(&mat_file);
        MPI_Finalize();
        return 1;
    }

    MPI_Offset mat_offset = sizeof(int) + rank * block_size * matrix_dim * sizeof(double);
    MPI_File_seek(mat_file, mat_offset, MPI_SEEK_SET);
    MPI_File_read(mat_file, local_matrix, block_size * matrix_dim, MPI_DOUBLE, MPI_STATUS_IGNORE);

    // Open and read the vector file
    MPI_File vec_file;
    MPI_File_open(MPI_COMM_WORLD, VECTOR_FILE, MPI_MODE_RDONLY, MPI_INFO_NULL, &vec_file);

    int vector_len;
    MPI_File_read(vec_file, &vector_len, 1, MPI_INT, MPI_STATUS_IGNORE);
    // Check if the vector length is 20
    if (vector_len != 20) {
        printf("Vector length error\n");
        MPI_File_close(&vec_file);
        MPI_File_close(&mat_file);
        MPI_Finalize();
        return 1;
    }

    double* vector = (double*)malloc(vector_len * sizeof(double));
    // Check if memory allocation for the vector is successful
    if (vector == NULL) {
        printf("Memory error for vector\n");
        MPI_File_close(&vec_file);
        MPI_File_close(&mat_file);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        MPI_File_seek(vec_file, sizeof(int), MPI_SEEK_SET);
        MPI_File_read(vec_file, vector, vector_len, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }

    // Broadcast the vector to all processes
    MPI_Bcast(vector, vector_len, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calculate the matrix-vector product Ax
    double* local_result = (double*)malloc(block_size * sizeof(double));
    // Check if memory allocation for the local result is successful
    if (local_result == NULL) {
        printf("Memory error for local result\n");
        free(local_matrix);
        free(vector);
        MPI_File_close(&vec_file);
        MPI_File_close(&mat_file);
        MPI_Finalize();
        return 1;
    }

    for (int i = 0; i < block_size; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < matrix_dim; j++) {
            local_result[i] += local_matrix[i * matrix_dim + j] * vector[j];
        }
    }

    double* global_result = NULL;
    if (rank == 0) {
        global_result = (double*)malloc(20 * sizeof(double));
        // Check if memory allocation for the global result is successful
        if (global_result == NULL) {
            printf("Memory error for global result\n");
            free(local_matrix);
            free(vector);
            free(local_result);
            MPI_File_close(&vec_file);
            MPI_File_close(&mat_file);
            MPI_Finalize();
            return 1;
        }
    }

    // Gather all local results to the root process
    MPI_Gather(local_result, block_size, MPI_DOUBLE, global_result, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print the complete result in the root process
    if (rank == 0) {
        printf("Matrix-vector product result:\n");
        for (int i = 0; i < 20; i++) {
            printf("%d\n", (int)global_result[i]);
        }
        free(global_result);
    }

    // Clean up resources
    free(local_matrix);
    free(vector);
    free(local_result);

    MPI_File_close(&mat_file);
    MPI_File_close(&vec_file);
    MPI_Finalize();
    return 0;
}    