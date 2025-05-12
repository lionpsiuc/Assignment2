#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "jacobi2d.h"
#include "poisson1d.h"
#include "write_grid.h"

#define maxit 2000 // Maximum iterations

int main(int argc, char** argv) {
  double a[maxn][maxn], b[maxn][maxn], f[maxn][maxn]; // Local grid
  double global_a[maxn][maxn]; // Global grid (used only by rank 0)
  int    nx, ny;
  int    myid, nprocs;
  Cart2D cart2d;
  int    it;
  double glob_diff, glob_max_error;
  double ldiff, local_max_error;
  double t1, t2;
  double tol = 1.0E-11;
  char   name[1024];
  int    namelen;
  int    use_overlapped = 0; // Overlapped computation not implemented for RMA

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  MPI_Get_processor_name(name, &namelen);
  printf("(myid %d): running on node: %s\n", myid, name);

  if (myid == 0) {
    // Default grid size is 31 as required in the problem
    nx = 31;

    // Parse command line arguments
    if (argc >= 2) {
      nx = atoi(argv[1]);
    }

    if (argc >= 3) {
      use_overlapped = atoi(argv[2]);
    }

    if (nx > maxn - 2) {
      fprintf(stderr, "Grid size too large\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    printf("Grid size: %d x %d\n", nx, nx);
    printf("Using RMA communication with PSCW synchronization\n");

    if (use_overlapped) {
      printf("WARNING: Overlapped computation not implemented for RMA PSCW\n");
      use_overlapped = 0;
    }

    printf("Using standard computation\n");
  }

  // Broadcast configuration parameters
  MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&use_overlapped, 1, MPI_INT, 0, MPI_COMM_WORLD);

  ny = nx;

  // Zero-initialize arrays
  for (int i = 0; i < maxn; i++) {
    for (int j = 0; j < maxn; j++) {
      a[i][j] = 0.0;
      b[i][j] = 0.0;
      f[i][j] = 0.0;
      if (myid == 0) {
        global_a[i][j] = 0.0;
      }
    }
  }

  // Initialize the 2D Cartesian topology
  init_cart2d(&cart2d, MPI_COMM_WORLD, nx, ny, a);

  // Set boundary conditions
  init_grid2d(a, f, nx, ny, &cart2d);
  init_grid2d(b, f, nx, ny, &cart2d);

  share_remote_offsets(&cart2d);

  // Start timer
  MPI_Barrier(cart2d.cart_comm);
  t1 = MPI_Wtime();

  // Main iteration loop
  glob_diff = 1000.0;
  for (it = 0; it < maxit && glob_diff > tol; it++) {
    // Update grid using Jacobi iteration with PSCW synchronization
    exchange_ghost_rma_pscw(a, nx, ny, &cart2d);
    sweep2d(a, f, b, nx, ny, &cart2d);

    exchange_ghost_rma_pscw(b, nx, ny, &cart2d);
    sweep2d(b, f, a, nx, ny, &cart2d);

    // Check convergence
    ldiff = grid_diff2d(a, b, nx, ny, &cart2d);
    MPI_Allreduce(&ldiff, &glob_diff, 1, MPI_DOUBLE, MPI_SUM, cart2d.cart_comm);

    if (myid == 0 && (it % 100 == 0 || it == maxit - 1)) {
      printf("(Iteration %d) Global difference: %e\n", it, glob_diff);
    }
  }

  // Stop timer
  t2 = MPI_Wtime();

  // Print timing and convergence information
  if (myid == 0) {
    if (it >= maxit) {
      printf("Maximum iterations (%d) reached.\n", maxit);
    } else {
      printf("Converged after %d iterations.\n", it);
    }
    printf("Computation time: %lf seconds\n", t2 - t1);
  }

  // Calculate error between numerical and analytical solutions
  local_max_error = calculate_error2d(a, nx, ny, &cart2d);
  MPI_Reduce(&local_max_error, &glob_max_error, 1, MPI_DOUBLE, MPI_MAX, 0,
             cart2d.cart_comm);

  if (myid == 0) {
    printf("\nMaximum error between numerical and analytic solutions: %e\n",
           glob_max_error);
  }

  // Gather the solution to rank 0
  gather_grid2d(global_a, a, nx, ny, &cart2d);

  // Print and save results (rank 0 only)
  if (myid == 0) {
    char filename[100];
    sprintf(filename, "poisson2d_sol_rma_pscw_nx%d_np%d.txt", nx, nprocs);

    // Write the solution to file
    write_grid(filename, global_a, nx, ny);
    printf("Grid written to file: %s\n", filename);
    printf("Solution written to file: %s\n", filename);

    // Global Error Verification
    double global_max_error = 0.0;
    double h                = 1.0 / (nx + 1);

    for (int i = 1; i <= nx; i++) {
      for (int j = 1; j <= ny; j++) {
        double x     = j * h;
        double y     = i * h;
        double exact = analytical_solution(x, y);
        double error = fabs(global_a[i][j] - exact);

        if (error > global_max_error) {
          global_max_error = error;
        }
      }
    }

    printf("\nGlobal maximum error after gathering: %e\n", global_max_error);
    printf("Compare with distributed error calculation: %e\n", glob_max_error);
  }

  // Free resources
  free_cart2d(&cart2d);
  MPI_Finalize();
  return 0;
}
