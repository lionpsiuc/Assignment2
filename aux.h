#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define maxn (31 + 2)

void init_full_grids(double a[][maxn], double b[][maxn], double f[][maxn]);

void init_twod(double a[][maxn], double b[][maxn], double f[][maxn], int nx,
               int ny __attribute__((unused)), int row_s, int row_e, int col_s,
               int col_e);

int MPE_Decomp2d(int nrows, int ncols, int rank __attribute__((unused)),
                 int* coords, int* row_s, int* row_e, int* col_s, int* col_e,
                 int* dims);

void GatherGrid2D(double global_grid[][maxn], double a[][maxn], int row_s,
                  int row_e, int col_s, int col_e, int nx, int ny, int myid,
                  int nprocs, int* row_s_vals, int* row_e_vals, int* col_s_vals,
                  int* col_e_vals, MPI_Comm comm);

void write_grid(char* filename, double a[][maxn],
                int nx __attribute__((unused)), int ny __attribute__((unused)),
                int rank, int row_s, int row_e, int col_s, int col_e,
                int write_to_stdout);

double griddiff2d(double a[][maxn], double b[][maxn],
                  int nx __attribute__((unused)), int row_s, int row_e,
                  int col_s, int col_e);

void sweep2d(double a[][maxn], double f[][maxn], int nx, int row_s, int row_e,
             int col_s, int col_e, double b[][maxn]);

void exchang2d_rma_fence(double x[][maxn], int row_s, int row_e, int col_s,
                         int col_e, int nbrleft, int nbrright, int nbrup,
                         int nbrdown, MPI_Datatype row_type, MPI_Win win);

void exchang2d_rma_pscw(double x[][maxn], int row_s, int row_e, int col_s,
                        int col_e, int nbrleft, int nbrright, int nbrup,
                        int nbrdown, MPI_Datatype row_type, MPI_Win win,
                        MPI_Group group);
