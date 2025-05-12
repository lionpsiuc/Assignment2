#include "aux.h"

void init_full_grids(double a[][maxn], double b[][maxn], double f[][maxn]) {
  const double junkval = -5;
  for (int i = 0; i < maxn; i++) {
    for (int j = 0; j < maxn; j++) {
      a[i][j] = junkval;
      b[i][j] = junkval;
      f[i][j] = junkval;
    }
  }
}

void init_twod(double a[][maxn], double b[][maxn], double f[][maxn], int nx,
               int ny __attribute__((unused)), int row_s, int row_e, int col_s,
               int col_e) {
  double h = 1.0 / ((double) (nx + 1)); // Grid spacing

  // Set everything to zero first
  for (int i = col_s - 1; i <= col_e + 1; i++) {
    for (int j = row_s - 1; j <= row_e + 1; j++) {
      a[i][j] = 0.0;
      b[i][j] = 0.0;
      f[i][j] = 0.0;
    }
  }

  if (row_e == nx) {
    for (int i = col_s; i <= col_e; i++) {
      double x     = i * h; // Transform to coordinate system
      a[i][nx + 1] = 1.0 / ((1.0 + x) * (1.0 + x) + 1.0);
      b[i][nx + 1] = 1.0 / ((1.0 + x) * (1.0 + x) + 1.0);
    }
  }
  if (row_s == 1) {
    for (int i = col_s; i <= col_e; i++) {
      a[i][0] = 0.0;
      b[i][0] = 0.0;
    }
  }
  if (col_s == 1) {
    for (int j = row_s; j <= row_e; j++) {
      double y = j * h; // Transform to coordinate system
      a[0][j]  = y / (1.0 + y * y);
      b[0][j]  = y / (1.0 + y * y);
    }
  }
  if (col_e == nx) {
    for (int j = row_s; j <= row_e; j++) {
      double y     = j * h; // Transform to coordinate system
      a[nx + 1][j] = y / (4.0 + y * y);
      b[nx + 1][j] = y / (4.0 + y * y);
    }
  }
}

int MPE_Decomp2d(int nrows, int ncols, int rank __attribute__((unused)),
                 int* coords, int* row_s, int* row_e, int* col_s, int* col_e,
                 int* dims) {
  int rows_per_proc, cols_per_proc, row_deficit, col_deficit;
  rows_per_proc = nrows / dims[0];
  row_deficit   = nrows % dims[0];
  *row_s        = coords[0] * rows_per_proc +
           ((coords[0] < row_deficit) ? coords[0] : row_deficit) + 1;
  if (coords[0] < row_deficit)
    rows_per_proc++;
  *row_e = *row_s + rows_per_proc - 1;
  if (*row_e > nrows || coords[0] == dims[0] - 1)
    *row_e = nrows;
  int total_rows_assigned = *row_e - *row_s + 1;
  *row_e                  = nrows - (*row_s - 1);
  *row_s                  = *row_e - total_rows_assigned + 1;
  cols_per_proc           = ncols / dims[1];
  col_deficit             = ncols % dims[1];
  *col_s                  = coords[1] * cols_per_proc +
           ((coords[1] < col_deficit) ? coords[1] : col_deficit) + 1;
  if (coords[1] < col_deficit)
    cols_per_proc++;
  *col_e = *col_s + cols_per_proc - 1;
  if (*col_e > ncols || coords[1] == dims[1] - 1)
    *col_e = ncols;
  return MPI_SUCCESS;
}

void GatherGrid2D(double global_grid[][maxn], double a[][maxn], int row_s,
                  int row_e, int col_s, int col_e, int nx, int ny, int myid,
                  int nprocs, int* row_s_vals, int* row_e_vals, int* col_s_vals,
                  int* col_e_vals, MPI_Comm comm) {
  if (myid == 0) {

    // Initialize the global grid first
    for (int i = 0; i < maxn; i++) {
      for (int j = 0; j < maxn; j++) {
        global_grid[i][j] = 0.0;
      }
    }

    // Copy local data from root process
    for (int i = col_s; i <= col_e; i++) {
      for (int j = row_s; j <= row_e; j++) {
        global_grid[i][j] = a[i][j];
      }
    }

    double h = 1.0 / ((double) (nx + 1)); // Grid spacing

    // Set the top boundary where u(x,1)=1/((1+x)^2+1)
    for (int i = 0; i <= nx + 1; i++) {
      double x               = i * h;
      global_grid[i][ny + 1] = 1.0 / ((1.0 + x) * (1.0 + x) + 1.0);
    }

    // Set the left boundary where u(0,y)=y/(1+y^2)
    for (int j = 0; j <= ny + 1; j++) {
      double y = j * h;
      if (j == 0 || (1.0 + y * y) == 0.0) { // Protect against division by zero
        global_grid[0][j] = 0.0;
      } else {
        global_grid[0][j] = y / (1.0 + y * y);
      }
    }

    // Set the right boundary where u(1,y)=y/(4+y^2)
    for (int j = 0; j <= ny + 1; j++) {
      double y = j * h;
      if (j == 0 || (4.0 + y * y) == 0.0) { // Protect against division by zero
        global_grid[nx + 1][j] = 0.0;
      } else {
        global_grid[nx + 1][j] = y / (4.0 + y * y);
      }
    }
  }

  // Synchronize before data exchange
  MPI_Barrier(comm);

  // Receive data into root process from other processes
  if (myid != 0) {
    int local_rows = row_e - row_s + 1;
    for (int col = col_s; col <= col_e; col++) {
      int tag = 10000 + myid * 100 + col; // Use a unique tag
      MPI_Send(&a[col][row_s], local_rows, MPI_DOUBLE, 0, tag, comm);
    }
  } else { // Root process receives from all other processes
    for (int p = 1; p < nprocs; p++) {
      int p_row_s      = row_s_vals[p];
      int p_row_e      = row_e_vals[p];
      int p_col_s      = col_s_vals[p];
      int p_col_e      = col_e_vals[p];
      int p_local_rows = p_row_e - p_row_s + 1;
      for (int col = p_col_s; col <= p_col_e; col++) {
        MPI_Status status;
        int        tag = 10000 + p * 100 + col; // Use a unique tag
        MPI_Recv(&global_grid[col][p_row_s], p_local_rows, MPI_DOUBLE, p, tag,
                 comm, &status);
      }
    }

    // Message to state when the function has completed its task
    printf("Gathering complete\n");
  }
}

void write_grid(char* filename, double a[][maxn],
                int nx __attribute__((unused)), int ny __attribute__((unused)),
                int rank, int row_s, int row_e, int col_s, int col_e,
                int write_to_stdout) {

  // Create filename with extension
  char full_filename[256];
  sprintf(full_filename, "%s.txt", filename);

  // Open file for writing
  FILE* file = fopen(full_filename, "w");
  if (!file) {
    fprintf(stderr, "Error opening file %s for writing\n", full_filename);
    return;
  }

  // Write to file in mesh/grid format; note that each row is a y-coordinate
  // whereas each column is an x-coordinate
  for (int j = row_e; j >= row_s; j--) {
    for (int i = col_s; i <= col_e; i++) {
      fprintf(file, "%.6lf ", a[i][j]);
    }
    fprintf(file, "\n");
  }

  fclose(file);

  // Write to terminal if requested
  if (write_to_stdout) {
    printf("Grid for process %d\n", rank);
    for (int j = row_e; j >= row_s; j--) {
      for (int i = col_s; i <= col_e; i++) {
        printf("%.6lf ", a[i][j]);
      }
      printf("\n");
    }
  }
}

double griddiff2d(double a[][maxn], double b[][maxn],
                  int nx __attribute__((unused)), int row_s, int row_e,
                  int col_s, int col_e) {
  double sum = 0.0;
  double tmp;
  for (int i = col_s; i <= col_e; i++) {
    for (int j = row_s; j <= row_e; j++) {
      tmp = (a[i][j] - b[i][j]);
      sum = sum + tmp * tmp;
    }
  }
  return sum;
}

void sweep2d(double a[][maxn], double f[][maxn], int nx, int row_s, int row_e,
             int col_s, int col_e, double b[][maxn]) {
  double h = 1.0 / ((double) (nx + 1)); // Grid spacing
  for (int i = col_s; i <= col_e; i++) {
    for (int j = row_s; j <= row_e; j++) {
      b[i][j] = 0.25 * (a[i - 1][j] + a[i + 1][j] + a[i][j + 1] + a[i][j - 1] -
                        h * h * f[i][j]);
    }
  }
}

void exchang2d_rma_fence(double x[][maxn], int row_s, int row_e, int col_s,
                         int col_e, int nbrleft, int nbrright, int nbrup,
                         int nbrdown, MPI_Datatype row_type, MPI_Win win) {
  int lny = row_e - row_s + 1; // Number of rows in local domain

  // Start the RMA access epoch
  MPI_Win_fence(0, win);

  // Get their (left neighbor) rightmost column into our left ghost column
  if (nbrleft != MPI_PROC_NULL) {

    // We want to get the data at column (col_s - 1) from our left neighbor
    MPI_Aint displacement = (MPI_Aint) (col_s - 1) * maxn + row_s;
    MPI_Get(&x[col_s - 1][row_s], lny, MPI_DOUBLE, nbrleft, displacement, lny,
            MPI_DOUBLE, win);
  }

  // Get their (right neighbor) leftmost column into our right ghost column
  if (nbrright != MPI_PROC_NULL) {

    // We want to get the data at column (col_e + 1) from our right neighbor
    MPI_Aint displacement = (MPI_Aint) (col_e + 1) * maxn + row_s;
    MPI_Get(&x[col_e + 1][row_s], lny, MPI_DOUBLE, nbrright, displacement, lny,
            MPI_DOUBLE, win);
  }

  // Get their (lower neighbor) topmost row into our bottom ghost row
  if (nbrdown != MPI_PROC_NULL) {

    // We want to get the data at row (row_s - 1) from our lower neighbor
    MPI_Aint displacement = (MPI_Aint) col_s * maxn + (row_s - 1);
    MPI_Get(&x[col_s][row_s - 1], 1, row_type, nbrdown, displacement, 1,
            row_type, win);
  }

  // Get their (upper neighbor) bottommost row into our top ghost row
  if (nbrup != MPI_PROC_NULL) {

    // We want to get the data at row (row_e + 1) from our upper neighbor
    MPI_Aint displacement = (MPI_Aint) col_s * maxn + (row_e + 1);
    MPI_Get(&x[col_s][row_e + 1], 1, row_type, nbrup, displacement, 1, row_type,
            win);
  }

  // End the RMA access epoch
  MPI_Win_fence(0, win);
}

void exchang2d_rma_pscw(double x[][maxn], int row_s, int row_e, int col_s,
                        int col_e, int nbrleft, int nbrright, int nbrup,
                        int nbrdown, MPI_Datatype row_type, MPI_Win win,
                        MPI_Group group) {
  int lny = row_e - row_s + 1; // Number of rows in local domain

  // Arrays to hold the ranks of processes we communicate with
  int access_neighbors[4];   // Processes we will read from
  int exposure_neighbors[4]; // Processes that will read from us
  int num_access   = 0;      // Counter for access neighbors
  int num_exposure = 0;      // Counter for exposure neighbors

  // Build the list of neighbors we need to access
  if (nbrleft != MPI_PROC_NULL)
    access_neighbors[num_access++] = nbrleft;
  if (nbrright != MPI_PROC_NULL)
    access_neighbors[num_access++] = nbrright;
  if (nbrup != MPI_PROC_NULL)
    access_neighbors[num_access++] = nbrup;
  if (nbrdown != MPI_PROC_NULL)
    access_neighbors[num_access++] = nbrdown;

  // The processes we read from are the same ones that read from us
  num_exposure = num_access;
  for (int i = 0; i < num_access; i++) {
    exposure_neighbors[i] = access_neighbors[i];
  }

  // Groups for synchronization
  MPI_Group access_group, exposure_group;

  // Create the access group
  if (num_access > 0) {
    MPI_Group_incl(group, num_access, access_neighbors, &access_group);
  }

  // Create the exposure group
  if (num_exposure > 0) {
    MPI_Group_incl(group, num_exposure, exposure_neighbors, &exposure_group);
  }

  // Post our window for exposure
  if (num_exposure > 0) {
    MPI_Win_post(exposure_group, 0, win);
  }

  // Start our access epoch
  if (num_access > 0) {
    MPI_Win_start(access_group, 0, win);
  }

  // Get their (left neighbor) rightmost column into our left ghost column
  if (nbrleft != MPI_PROC_NULL) {

    // Calculate the column position we want to access in the neighbor's memory
    int target_col = col_s - 1;

    // Calculate the memory displacement using row-major addressing
    MPI_Aint displacement = (MPI_Aint) target_col * maxn + row_s;

    MPI_Get(&x[col_s - 1][row_s], lny, MPI_DOUBLE, nbrleft, displacement, lny,
            MPI_DOUBLE, win);
  }

  // Get their (right neighbor) leftmost column into our right ghost column
  if (nbrright != MPI_PROC_NULL) {
    int      target_col   = col_e + 1;
    MPI_Aint displacement = (MPI_Aint) target_col * maxn + row_s;
    MPI_Get(&x[col_e + 1][row_s], lny, MPI_DOUBLE, nbrright, displacement, lny,
            MPI_DOUBLE, win);
  }

  // Get their (lower neighbor) topmost row into our bottom ghost row
  if (nbrdown != MPI_PROC_NULL) {
    int      target_row   = row_s - 1;
    MPI_Aint displacement = (MPI_Aint) col_s * maxn + target_row;
    MPI_Get(&x[col_s][row_s - 1], 1, row_type, nbrdown, displacement, 1,
            row_type, win);
  }

  // Get their (upper neighbor) bottommost row into our top ghost row
  if (nbrup != MPI_PROC_NULL) {
    int      target_row   = row_e + 1;
    MPI_Aint displacement = (MPI_Aint) col_s * maxn + target_row;
    MPI_Get(&x[col_s][row_e + 1], 1, row_type, nbrup, displacement, 1, row_type,
            win);
  }

  // Complete our access epoch
  if (num_access > 0) {
    MPI_Win_complete(win);
  }

  // Wait for our exposure to complete
  if (num_exposure > 0) {
    MPI_Win_wait(win);
  }

  // Free the groups we created
  if (num_access > 0)
    MPI_Group_free(&access_group);
  if (num_exposure > 0)
    MPI_Group_free(&exposure_group);
}
