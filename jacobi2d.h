#ifndef JACOBI2D_H
#define JACOBI2D_H

#include <mpi.h>
#include "poisson1d.h"

// Structure to hold 2D decomposition information
typedef struct
{
    int coords[2];      // 2D coordinates of this process
    int dims[2];        // Number of processes in each dimension
    int local_size[2];  // Size of the local subdomain (excluding ghost cells)
    int local_start[2]; // Global starting indices of this subdomain
    int local_end[2];   // Global ending indices of this subdomain

    // Neighbor ranks in the 2D topology
    int nbr_up;
    int nbr_down;
    int nbr_left;
    int nbr_right;

    // MPI_Datatype for column exchanges
    MPI_Datatype column_type;

    // MPI communicator with 2D Cartesian topology
    MPI_Comm cart_comm;

    // RMA window for the grid
    MPI_Win grid_win;

    // Groups for PSCW operations
    MPI_Group world_group;  // Group of all processes in the communicator
    MPI_Group post_group;   // Group of processes that will access my data (neighbors)
    MPI_Group start_group;  // Group of processes whose data I will access (neighbors)
    int *post_ranks;        // Ranks in post group
    int *start_ranks;       // Ranks in start group
    int post_count;         // Number of ranks in post group
    int start_count;        // Number of ranks in start group
} Cart2D;

// Initialize the 2D Cartesian topology
void init_cart2d(Cart2D *cart2d, MPI_Comm comm_old, int nx, int ny, double grid[][maxn]);

// Initialize the grid with boundary conditions
void init_grid2d(double grid[][maxn], double f[][maxn], int nx, int ny, Cart2D *cart2d);

// Exchange ghost cells using MPI_Sendrecv
void exchange_ghost_sendrecv(double grid[][maxn], int nx, int ny, Cart2D *cart2d);

// Exchange ghost cells using non-blocking MPI_Isend/MPI_Irecv
void exchange_ghost_nonblocking(double grid[][maxn], int nx, int ny, Cart2D *cart2d);

// Exchange ghost cells using RMA with MPI_Win_fence
void exchange_ghost_rma_fence(double grid[][maxn], int nx, int ny, Cart2D *cart2d);

void share_remote_offsets(Cart2D *cart2d);

// Exchange ghost cells using RMA with MPI_Win_post/start/complete/wait
void exchange_ghost_rma_pscw(double grid[][maxn], int nx, int ny, Cart2D *cart2d);

// Perform one sweep of Jacobi iteration on the 2D grid
void sweep2d(double a[][maxn], double f[][maxn], double b[][maxn], int nx, int ny, Cart2D *cart2d);

// Perform one sweep of Jacobi iteration with overlapping communication and computation
void sweep2d_overlapped(double a[][maxn], double f[][maxn], double b[][maxn], int nx, int ny, Cart2D *cart2d, MPI_Request reqs[]);

// Calculate local grid difference (for convergence check)
double grid_diff2d(double a[][maxn], double b[][maxn], int nx, int ny, Cart2D *cart2d);

// Gather the distributed solution to rank 0
void gather_grid2d(double global_grid[][maxn], double local_grid[][maxn], int nx, int ny, Cart2D *cart2d);

// Calculate the error between numerical and analytical solutions
double calculate_error2d(double a[][maxn], int nx, int ny, Cart2D *cart2d);

// Analytical solution for the problem
double analytical_solution(double x, double y);

// Free resources used by Cart2D
void free_cart2d(Cart2D *cart2d);

#endif /* JACOBI2D_H */
