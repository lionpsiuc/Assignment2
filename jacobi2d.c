#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include "jacobi2d.h"
#include "poisson1d.h"

#define TAG_VERTICAL_EXCHANGE 100
#define TAG_HORIZONTAL_EXCHANGE 101

extern MPI_Aint offset_up, offset_down, offset_left, offset_right;

void init_cart2d(Cart2D *cart2d, MPI_Comm comm_old, int nx, int ny, double grid[][maxn])
{
    int nprocs, rank;
    MPI_Comm_size(comm_old, &nprocs);
    MPI_Comm_rank(comm_old, &rank);

    // Automatically determine dimensions for the 2D decomposition
    int dims[2] = {0, 0};
    MPI_Dims_create(nprocs, 2, dims);

    int periods[2] = {0, 0}; // Non-periodic boundaries
    int reorder = 1;         // Allow MPI to reorder ranks for better performance

    // Create the 2D Cartesian communicator
    MPI_Cart_create(comm_old, 2, dims, periods, reorder, &(cart2d->cart_comm));

    // Get the new rank and coordinates in the Cartesian topology
    int cart_rank;
    MPI_Comm_rank(cart2d->cart_comm, &cart_rank);
    MPI_Cart_coords(cart2d->cart_comm, cart_rank, 2, cart2d->coords);

    // Store the dimensions
    cart2d->dims[0] = dims[0];
    cart2d->dims[1] = dims[1];

    // Get the neighbor ranks in all directions
    MPI_Cart_shift(cart2d->cart_comm, 0, 1, &(cart2d->nbr_up), &(cart2d->nbr_down));
    MPI_Cart_shift(cart2d->cart_comm, 1, 1, &(cart2d->nbr_left), &(cart2d->nbr_right));

    // Calculate local domain size and indices
    // Rows (y direction)
    int rows_per_proc = (ny + cart2d->dims[0] - 1) / cart2d->dims[0]; // Ceiling division
    cart2d->local_size[0] = rows_per_proc;
    // Adjust for the last process in the row
    if (cart2d->coords[0] == cart2d->dims[0] - 1)
    {
        cart2d->local_size[0] = ny - (cart2d->dims[0] - 1) * rows_per_proc;
        if (cart2d->local_size[0] <= 0)
            cart2d->local_size[0] = 1; // Ensure at least one point
    }

    // Columns (x direction)
    int cols_per_proc = (nx + cart2d->dims[1] - 1) / cart2d->dims[1]; // Ceiling division
    cart2d->local_size[1] = cols_per_proc;
    // Adjust for the last process in the column
    if (cart2d->coords[1] == cart2d->dims[1] - 1)
    {
        cart2d->local_size[1] = nx - (cart2d->dims[1] - 1) * cols_per_proc;
        if (cart2d->local_size[1] <= 0)
            cart2d->local_size[1] = 1; // Ensure at least one point
    }

    // Calculate global indices for this process
    cart2d->local_start[0] = cart2d->coords[0] * rows_per_proc + 1; // Add 1 because indices start at 1
    cart2d->local_end[0] = cart2d->local_start[0] + cart2d->local_size[0] - 1;

    cart2d->local_start[1] = cart2d->coords[1] * cols_per_proc + 1; // Add 1 because indices start at 1
    cart2d->local_end[1] = cart2d->local_start[1] + cart2d->local_size[1] - 1;

    // Create MPI datatype for column exchange
    MPI_Type_vector(cart2d->local_size[0], 1, maxn, MPI_DOUBLE, &(cart2d->column_type));
    MPI_Type_commit(&(cart2d->column_type));

    // Create RMA window for the grid
    MPI_Win_create(&grid[0][0], maxn * maxn * sizeof(double), sizeof(double),
                   MPI_INFO_NULL, cart2d->cart_comm, &(cart2d->grid_win));

    // Print domain decomposition info
    printf("Process %d (%d,%d): Domain [%d:%d, %d:%d], Size [%d,%d], Neighbors: U=%d, D=%d, L=%d, R=%d\n",
           cart_rank, cart2d->coords[0], cart2d->coords[1],
           cart2d->local_start[0], cart2d->local_end[0],
           cart2d->local_start[1], cart2d->local_end[1],
           cart2d->local_size[0], cart2d->local_size[1],
           cart2d->nbr_up, cart2d->nbr_down, cart2d->nbr_left, cart2d->nbr_right);
}

// Initialize the grid with boundary conditions
void init_grid2d(double grid[][maxn], double f[][maxn], int nx, int ny, Cart2D *cart2d)
{
    int i, j;
    double h = 1.0 / (nx + 1); // Grid spacing
    double x, y;

    // Initialize all grid points to zero
    for (i = 0; i < maxn; i++)
    {
        for (j = 0; j < maxn; j++)
        {
            grid[i][j] = 0.0;
            f[i][j] = 0.0; // f = 0 for Laplace equation
        }
    }

    // Set boundary values based on global coordinates
    // Bottom boundary: u(x,0) = 0
    if (cart2d->coords[0] == 0)
    {
        for (j = 1; j <= cart2d->local_size[1]; j++)
        {
            grid[0][j] = 0.0;
        }
    }

    // Top boundary: u(x,1) = 1/((1+x)^2 + 1)
    // Top boundary: u(x,1) = 1/((1+x)^2 + 1)
    if (cart2d->coords[0] == cart2d->dims[0] - 1)
    {
        for (j = 1; j <= cart2d->local_size[1]; j++)
        {
            // Calculate the global physical x-coordinate
            int global_j = cart2d->local_start[1] + j - 1;
            // Convert global index to physical coordinate
            x = global_j * h;
            grid[cart2d->local_size[0] + 1][j] = 1.0 / ((1.0 + x) * (1.0 + x) + 1.0);
        }
    }

    // Left boundary: u(0,y) = y/(1+y^2)
    if (cart2d->coords[1] == 0)
    {
        for (i = 1; i <= cart2d->local_size[0]; i++)
        {
            int global_i = cart2d->local_start[0] + i - 1;
            y = global_i * h;
            grid[i][0] = y / (1.0 + y * y);
        }
    }

    // Right boundary: u(1,y) = y/(4+y^2)
    if (cart2d->coords[1] == cart2d->dims[1] - 1)
    {
        for (i = 1; i <= cart2d->local_size[0]; i++)
        {
            int global_i = cart2d->local_start[0] + i - 1;
            y = global_i * h;
            grid[i][cart2d->local_size[1] + 1] = y / (4.0 + y * y);
        }
    }
}

// Exchange ghost cells using MPI_Sendrecv
void exchange_ghost_sendrecv(double grid[][maxn], int nx, int ny, Cart2D *cart2d)
{
    // int i, j;
    MPI_Status status;

    // Exchange with top neighbor (rows)
    MPI_Sendrecv(
        &grid[1][1], cart2d->local_size[1], MPI_DOUBLE, cart2d->nbr_up, 0,
        &grid[0][1], cart2d->local_size[1], MPI_DOUBLE, cart2d->nbr_up, 0,
        cart2d->cart_comm, &status);

    // Exchange with bottom neighbor (rows)
    MPI_Sendrecv(
        &grid[cart2d->local_size[0]][1], cart2d->local_size[1], MPI_DOUBLE, cart2d->nbr_down, 0,
        &grid[cart2d->local_size[0] + 1][1], cart2d->local_size[1], MPI_DOUBLE, cart2d->nbr_down, 0,
        cart2d->cart_comm, &status);

    // Exchange with left neighbor (columns)
    MPI_Sendrecv(
        &grid[1][1], 1, cart2d->column_type, cart2d->nbr_left, 0,
        &grid[1][0], 1, cart2d->column_type, cart2d->nbr_left, 0,
        cart2d->cart_comm, &status);

    // Exchange with right neighbor (columns)
    MPI_Sendrecv(
        &grid[1][cart2d->local_size[1]], 1, cart2d->column_type, cart2d->nbr_right, 0,
        &grid[1][cart2d->local_size[1] + 1], 1, cart2d->column_type, cart2d->nbr_right, 0,
        cart2d->cart_comm, &status);
}

// Exchange ghost cells using non-blocking MPI_Isend/MPI_Irecv
void exchange_ghost_nonblocking(double grid[][maxn], int nx, int ny, Cart2D *cart2d)
{
    MPI_Request reqs[8]; // 8 requests: 4 sends and 4 receives
    int req_count = 0;

    // Receive from top
    if (cart2d->nbr_up != MPI_PROC_NULL)
    {
        MPI_Irecv(&grid[0][1], cart2d->local_size[1], MPI_DOUBLE,
                  cart2d->nbr_up, 0, cart2d->cart_comm, &reqs[req_count++]);
    }

    // Receive from bottom
    if (cart2d->nbr_down != MPI_PROC_NULL)
    {
        MPI_Irecv(&grid[cart2d->local_size[0] + 1][1], cart2d->local_size[1], MPI_DOUBLE,
                  cart2d->nbr_down, 0, cart2d->cart_comm, &reqs[req_count++]);
    }

    // Receive from left
    if (cart2d->nbr_left != MPI_PROC_NULL)
    {
        MPI_Irecv(&grid[1][0], 1, cart2d->column_type,
                  cart2d->nbr_left, 0, cart2d->cart_comm, &reqs[req_count++]);
    }

    // Receive from right
    if (cart2d->nbr_right != MPI_PROC_NULL)
    {
        MPI_Irecv(&grid[1][cart2d->local_size[1] + 1], 1, cart2d->column_type,
                  cart2d->nbr_right, 0, cart2d->cart_comm, &reqs[req_count++]);
    }

    // Send to top
    if (cart2d->nbr_up != MPI_PROC_NULL)
    {
        MPI_Isend(&grid[1][1], cart2d->local_size[1], MPI_DOUBLE,
                  cart2d->nbr_up, 0, cart2d->cart_comm, &reqs[req_count++]);
    }

    // Send to bottom
    if (cart2d->nbr_down != MPI_PROC_NULL)
    {
        MPI_Isend(&grid[cart2d->local_size[0]][1], cart2d->local_size[1], MPI_DOUBLE,
                  cart2d->nbr_down, 0, cart2d->cart_comm, &reqs[req_count++]);
    }

    // Send to left
    if (cart2d->nbr_left != MPI_PROC_NULL)
    {
        MPI_Isend(&grid[1][1], 1, cart2d->column_type,
                  cart2d->nbr_left, 0, cart2d->cart_comm, &reqs[req_count++]);
    }

    // Send to right
    if (cart2d->nbr_right != MPI_PROC_NULL)
    {
        MPI_Isend(&grid[1][cart2d->local_size[1]], 1, cart2d->column_type,
                  cart2d->nbr_right, 0, cart2d->cart_comm, &reqs[req_count++]);
    }

    // Wait for all communications to complete
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
}

// Perform one sweep of Jacobi iteration on the 2D grid
void sweep2d(double a[][maxn], double f[][maxn], double b[][maxn], int nx, int ny, Cart2D *cart2d)
{
    int i, j;
    double h = 1.0 / (nx + 1);

    // Perform Jacobi update for internal points
    for (i = 1; i <= cart2d->local_size[0]; i++)
    {
        for (j = 1; j <= cart2d->local_size[1]; j++)
        {
            b[i][j] = 0.25 * (a[i - 1][j] + a[i + 1][j] + a[i][j - 1] + a[i][j + 1] - h * h * f[i][j]);
        }
    }
}

void exchange_ghost_rma_fence(double grid[][maxn], int nx, int ny, Cart2D *cart2d)
{
    // Start RMA access epoch with MPI_Win_fence
    // Using MPI_MODE_NOSTORE since we're not updating our local window with direct stores,
    // only through RMA operations
    MPI_Win_fence(MPI_MODE_NOSTORE, cart2d->grid_win);

    // Exchange with top neighbor (rows)
    if (cart2d->nbr_up != MPI_PROC_NULL)
    {
        // Get the bottom row from the top neighbor into my top ghost row
        MPI_Get(&grid[0][1], cart2d->local_size[1], MPI_DOUBLE,
                cart2d->nbr_up, (cart2d->local_size[0] * maxn) + 1,
                cart2d->local_size[1], MPI_DOUBLE, cart2d->grid_win);
    }

    // Exchange with bottom neighbor (rows)
    if (cart2d->nbr_down != MPI_PROC_NULL)
    {
        // Get the top row from the bottom neighbor into my bottom ghost row
        MPI_Get(&grid[cart2d->local_size[0] + 1][1], cart2d->local_size[1], MPI_DOUBLE,
                cart2d->nbr_down, 1 * maxn + 1,
                cart2d->local_size[1], MPI_DOUBLE, cart2d->grid_win);
    }

    // Exchange with left neighbor (columns)
    if (cart2d->nbr_left != MPI_PROC_NULL)
    {
        // Get the rightmost column from left neighbor into my left ghost column
        MPI_Get(&grid[1][0], 1, cart2d->column_type,
                cart2d->nbr_left, 1 * maxn + cart2d->local_size[1],
                1, cart2d->column_type, cart2d->grid_win);
    }

    // Exchange with right neighbor (columns)
    if (cart2d->nbr_right != MPI_PROC_NULL)
    {
        // Get the leftmost column from right neighbor into my right ghost column
        MPI_Get(&grid[1][cart2d->local_size[1] + 1], 1, cart2d->column_type,
                cart2d->nbr_right, 1 * maxn + 1,
                1, cart2d->column_type, cart2d->grid_win);
    }

    // Complete RMA access epoch
    MPI_Win_fence(0, cart2d->grid_win);
}

#define INDEX(i, j) ((i) * maxn + (j)

void share_remote_offsets(Cart2D *cart2d) {
  MPI_Status status;

  // Calculate displacements to my own data regions that my neighbors will access.
  // For rows (UP/DOWN neighbors):
  MPI_Aint my_first_row_disp = INDEX(1, 1); // Offset to the start of my first actual data row
  MPI_Aint my_last_row_disp  = INDEX(cart2d->local_size[0], 1); // Offset to the start of my last actual data row

  // For columns (LEFT/RIGHT neighbors), using MPI_Type_vector, the displacement is to the first element.
  MPI_Aint my_first_col_disp = INDEX(1, 1); // Offset to grid[1][1] (start of my first data column)
  MPI_Aint my_last_col_disp  = INDEX(1, cart2d->local_size[1]); // Offset to grid[1][local_size[1]] (start of my last data column)

  // --- Vertical Exchange (Up/Down Neighbors) ---
  // Current process P needs to get its nbr_up's LAST row. nbr_up needs to send its my_last_row_disp.
  // P will store this received offset in P.offset_up.
  // Symmetrically, P sends its my_first_row_disp to nbr_up (because nbr_up sees P as its nbr_down and needs P's first row).
  if (cart2d->nbr_up != MPI_PROC_NULL) {
      MPI_Sendrecv(&my_first_row_disp, 1, MPI_AINT, cart2d->nbr_up, TAG_VERTICAL_EXCHANGE,
                   &offset_up,        1, MPI_AINT, cart2d->nbr_up, TAG_VERTICAL_EXCHANGE,
                   cart2d->cart_comm, &status);
  }

  // Current process P needs to get its nbr_down's FIRST row. nbr_down needs to send its my_first_row_disp.
  // P will store this received offset in P.offset_down.
  // Symmetrically, P sends its my_last_row_disp to nbr_down (because nbr_down sees P as its nbr_up and needs P's last row).
  if (cart2d->nbr_down != MPI_PROC_NULL) {
      MPI_Sendrecv(&my_last_row_disp,  1, MPI_AINT, cart2d->nbr_down, TAG_VERTICAL_EXCHANGE,
                   &offset_down,      1, MPI_AINT, cart2d->nbr_down, TAG_VERTICAL_EXCHANGE,
                   cart2d->cart_comm, &status);
  }

  // --- Horizontal Exchange (Left/Right Neighbors) ---
  // Current process P needs to get its nbr_left's LAST column. nbr_left needs to send its my_last_col_disp.
  // P will store this received offset in P.offset_left.
  // Symmetrically, P sends its my_first_col_disp to nbr_left (as nbr_left sees P as its nbr_right).
  if (cart2d->nbr_left != MPI_PROC_NULL) {
      MPI_Sendrecv(&my_first_col_disp, 1, MPI_AINT, cart2d->nbr_left, TAG_HORIZONTAL_EXCHANGE,
                   &offset_left,      1, MPI_AINT, cart2d->nbr_left, TAG_HORIZONTAL_EXCHANGE,
                   cart2d->cart_comm, &status);
  }

  // Current process P needs to get its nbr_right's FIRST column. nbr_right needs to send its my_first_col_disp.
  // P will store this received offset in P.offset_right.
  // Symmetrically, P sends its my_last_col_disp to nbr_right (as nbr_right sees P as its nbr_left).
  if (cart2d->nbr_right != MPI_PROC_NULL) {
      MPI_Sendrecv(&my_last_col_disp,  1, MPI_AINT, cart2d->nbr_right, TAG_HORIZONTAL_EXCHANGE,
                   &offset_right,     1, MPI_AINT, cart2d->nbr_right, TAG_HORIZONTAL_EXCHANGE,
                   cart2d->cart_comm, &status);
  }
}

void exchange_ghost_rma_pscw(double grid[][maxn], int nx, int ny, Cart2D *cart2d)
{
    MPI_Group world_group, post_group, start_group;
    int post_ranks[4], start_ranks[4];
    int post_count = 0, start_count = 0;

    if (cart2d->nbr_up != MPI_PROC_NULL)
    {
        post_ranks[post_count] = cart2d->nbr_up;
        start_ranks[start_count] = cart2d->nbr_up;
        post_count++;
        start_count++;
    }
    if (cart2d->nbr_down != MPI_PROC_NULL)
    {
        post_ranks[post_count] = cart2d->nbr_down;
        start_ranks[start_count] = cart2d->nbr_down;
        post_count++;
        start_count++;
    }
    if (cart2d->nbr_left != MPI_PROC_NULL)
    {
        post_ranks[post_count] = cart2d->nbr_left;
        start_ranks[start_count] = cart2d->nbr_left;
        post_count++;
        start_count++;
    }
    if (cart2d->nbr_right != MPI_PROC_NULL)
    {
        post_ranks[post_count] = cart2d->nbr_right;
        start_ranks[start_count] = cart2d->nbr_right;
        post_count++;
        start_count++;
    }

    MPI_Comm_group(cart2d->cart_comm, &world_group);
    MPI_Group_incl(world_group, post_count, post_ranks, &post_group);
    MPI_Group_incl(world_group, start_count, start_ranks, &start_group);

    MPI_Win_post(post_group, 0, cart2d->grid_win);
    MPI_Win_start(start_group, 0, cart2d->grid_win);

    if (cart2d->nbr_up != MPI_PROC_NULL)
    {
        MPI_Get(&grid[0][1], cart2d->local_size[1], MPI_DOUBLE,
                cart2d->nbr_up, offset_up, cart2d->local_size[1], MPI_DOUBLE, cart2d->grid_win);
    }
    if (cart2d->nbr_down != MPI_PROC_NULL)
    {
        MPI_Get(&grid[cart2d->local_size[0] + 1][1], cart2d->local_size[1], MPI_DOUBLE,
                cart2d->nbr_down, offset_down, cart2d->local_size[1], MPI_DOUBLE, cart2d->grid_win);
    }
    if (cart2d->nbr_left != MPI_PROC_NULL)
    {
        MPI_Get(&grid[1][0], 1, cart2d->column_type,
                cart2d->nbr_left, offset_left, 1, cart2d->column_type, cart2d->grid_win);
    }
    if (cart2d->nbr_right != MPI_PROC_NULL)
    {
        MPI_Get(&grid[1][cart2d->local_size[1] + 1], 1, cart2d->column_type,
                cart2d->nbr_right, offset_right, 1, cart2d->column_type, cart2d->grid_win);
    }

    MPI_Win_complete(cart2d->grid_win);
    MPI_Win_wait(cart2d->grid_win);

    MPI_Group_free(&post_group);
    MPI_Group_free(&start_group);
    MPI_Group_free(&world_group);
    MPI_Barrier(cart2d->cart_comm);
}

// Perform one sweep of Jacobi iteration with overlapping communication and computation
void sweep2d_overlapped(double a[][maxn], double f[][maxn], double b[][maxn],
                        int nx, int ny, Cart2D *cart2d, MPI_Request reqs[])
{
    int i, j, req_count = 0;
    double h = 1.0 / (nx + 1);

    // Start non-blocking receives
    // Receive from top
    if (cart2d->nbr_up != MPI_PROC_NULL)
    {
        MPI_Irecv(&a[0][1], cart2d->local_size[1], MPI_DOUBLE,
                  cart2d->nbr_up, 0, cart2d->cart_comm, &reqs[req_count++]);
    }

    // Receive from bottom
    if (cart2d->nbr_down != MPI_PROC_NULL)
    {
        MPI_Irecv(&a[cart2d->local_size[0] + 1][1], cart2d->local_size[1], MPI_DOUBLE,
                  cart2d->nbr_down, 0, cart2d->cart_comm, &reqs[req_count++]);
    }

    // Receive from left
    if (cart2d->nbr_left != MPI_PROC_NULL)
    {
        MPI_Irecv(&a[1][0], 1, cart2d->column_type,
                  cart2d->nbr_left, 0, cart2d->cart_comm, &reqs[req_count++]);
    }

    // Receive from right
    if (cart2d->nbr_right != MPI_PROC_NULL)
    {
        MPI_Irecv(&a[1][cart2d->local_size[1] + 1], 1, cart2d->column_type,
                  cart2d->nbr_right, 0, cart2d->cart_comm, &reqs[req_count++]);
    }

    // Start non-blocking sends
    // Send to top
    if (cart2d->nbr_up != MPI_PROC_NULL)
    {
        MPI_Isend(&a[1][1], cart2d->local_size[1], MPI_DOUBLE,
                  cart2d->nbr_up, 0, cart2d->cart_comm, &reqs[req_count++]);
    }

    // Send to bottom
    if (cart2d->nbr_down != MPI_PROC_NULL)
    {
        MPI_Isend(&a[cart2d->local_size[0]][1], cart2d->local_size[1], MPI_DOUBLE,
                  cart2d->nbr_down, 0, cart2d->cart_comm, &reqs[req_count++]);
    }

    // Send to left
    if (cart2d->nbr_left != MPI_PROC_NULL)
    {
        MPI_Isend(&a[1][1], 1, cart2d->column_type,
                  cart2d->nbr_left, 0, cart2d->cart_comm, &reqs[req_count++]);
    }

    // Send to right
    if (cart2d->nbr_right != MPI_PROC_NULL)
    {
        MPI_Isend(&a[1][cart2d->local_size[1]], 1, cart2d->column_type,
                  cart2d->nbr_right, 0, cart2d->cart_comm, &reqs[req_count++]);
    }

    // Update interior points that don't depend on ghost cells
    // (i.e., points not adjacent to boundaries)
    for (i = 2; i < cart2d->local_size[0]; i++)
    {
        for (j = 2; j < cart2d->local_size[1]; j++)
        {
            b[i][j] = 0.25 * (a[i - 1][j] + a[i + 1][j] + a[i][j - 1] + a[i][j + 1] - h * h * f[i][j]);
        }
    }

    // Wait for all communications to complete
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

    // Update boundary points that depend on ghost cells
    // Top and bottom rows
    for (j = 1; j <= cart2d->local_size[1]; j++)
    {
        // Top row (i=1)
        b[1][j] = 0.25 * (a[0][j] + a[2][j] + a[1][j - 1] + a[1][j + 1] - h * h * f[1][j]);

        // Bottom row (i=local_size[0])
        b[cart2d->local_size[0]][j] = 0.25 * (a[cart2d->local_size[0] - 1][j] + a[cart2d->local_size[0] + 1][j] +
                                              a[cart2d->local_size[0]][j - 1] + a[cart2d->local_size[0]][j + 1] -
                                              h * h * f[cart2d->local_size[0]][j]);
    }

    // Left and right columns (excluding corners that are already done)
    for (i = 2; i < cart2d->local_size[0]; i++)
    {
        // Left column (j=1)
        b[i][1] = 0.25 * (a[i - 1][1] + a[i + 1][1] + a[i][0] + a[i][2] - h * h * f[i][1]);

        // Right column (j=local_size[1])
        b[i][cart2d->local_size[1]] = 0.25 * (a[i - 1][cart2d->local_size[1]] + a[i + 1][cart2d->local_size[1]] +
                                              a[i][cart2d->local_size[1] - 1] + a[i][cart2d->local_size[1] + 1] -
                                              h * h * f[i][cart2d->local_size[1]]);
    }
}

// Calculate local grid difference (for convergence check)
double grid_diff2d(double a[][maxn], double b[][maxn], int nx, int ny, Cart2D *cart2d)
{
    int i, j;
    double sum = 0.0;
    double tmp;

    for (i = 1; i <= cart2d->local_size[0]; i++)
    {
        for (j = 1; j <= cart2d->local_size[1]; j++)
        {
            tmp = a[i][j] - b[i][j];
            sum += tmp * tmp;
        }
    }

    return sum;
}

// Gather the distributed solution to rank 0
void gather_grid2d(double global_grid[][maxn], double local_grid[][maxn], int nx, int ny, Cart2D *cart2d)
{
    int i, j, proc, rank;
    MPI_Status status;

    // Get current rank
    MPI_Comm_rank(cart2d->cart_comm, &rank);

    if (rank == 0)
    {
        // Process 0 copies its own data first
        for (i = 1; i <= cart2d->local_size[0]; i++)
        {
            for (j = 1; j <= cart2d->local_size[1]; j++)
            {
                int global_i = cart2d->local_start[0] + i - 1;
                int global_j = cart2d->local_start[1] + j - 1;
                global_grid[global_i][global_j] = local_grid[i][j];
            }
        }

        // Receive data from other processes
        for (proc = 1; proc < cart2d->dims[0] * cart2d->dims[1]; proc++)
        {
            int coords[2];
            int local_size[2];
            int local_start[2];

            // Receive local domain information
            MPI_Recv(coords, 2, MPI_INT, proc, 0, cart2d->cart_comm, &status);
            MPI_Recv(local_size, 2, MPI_INT, proc, 1, cart2d->cart_comm, &status);
            MPI_Recv(local_start, 2, MPI_INT, proc, 2, cart2d->cart_comm, &status);

            // Receive grid data
            double buffer[maxn][maxn];
            MPI_Recv(&buffer[0][0], maxn * maxn, MPI_DOUBLE, proc, 3, cart2d->cart_comm, &status);

            // Copy received data to the global grid
            for (i = 1; i <= local_size[0]; i++)
            {
                for (j = 1; j <= local_size[1]; j++)
                {
                    int global_i = local_start[0] + i - 1;
                    int global_j = local_start[1] + j - 1;
                    global_grid[global_i][global_j] = buffer[i][j];
                }
            }
        }
        // After gathering all data, reset boundary conditions on the global grid
        double h = 1.0 / (nx + 1);

        // Bottom boundary: u(x,0) = 0
        for (int col = 0; col <= nx + 1; col++)
        {
            global_grid[0][col] = 0.0;
        }

        // Top boundary: u(x,1) = 1/((1+x)^2 + 1)
        for (int col = 0; col <= nx + 1; col++)
        {
            double x = col * h;
            global_grid[ny + 1][col] = 1.0 / ((1.0 + x) * (1.0 + x) + 1.0);
        }

        // Left boundary: u(0,y) = y/(1+y^2)
        for (int row = 1; row <= ny; row++)
        {
            double y = row * h;
            global_grid[row][0] = y / (1.0 + y * y);
        }

        // Right boundary: u(1,y) = y/(4+y^2)
        for (int row = 1; row <= ny; row++)
        {
            double y = row * h;
            global_grid[row][nx + 1] = y / (4.0 + y * y);
        }
    }
    else
    {
        // Send local domain information to rank 0
        MPI_Send(cart2d->coords, 2, MPI_INT, 0, 0, cart2d->cart_comm);
        MPI_Send(cart2d->local_size, 2, MPI_INT, 0, 1, cart2d->cart_comm);
        MPI_Send(cart2d->local_start, 2, MPI_INT, 0, 2, cart2d->cart_comm);

        // Send grid data to rank 0
        MPI_Send(&local_grid[0][0], maxn * maxn, MPI_DOUBLE, 0, 3, cart2d->cart_comm);
    }

    // Broadcast the global grid to all processes (optional, only if all need it)
    // MPI_Bcast(&global_grid[0][0], maxn*maxn, MPI_DOUBLE, 0, cart2d->cart_comm);
}

// Calculate the analytical solution for a given grid point
double analytical_solution(double x, double y)
{
    return y / ((1.0 + x) * (1.0 + x) + y * y);
}

// Calculate the error between numerical and analytical solutions
double calculate_error2d(double a[][maxn], int nx, int ny, Cart2D *cart2d)
{
    int i, j;
    double h = 1.0 / (nx + 1); // Grid spacing
    double x, y, exact_sol, error, max_error = 0.0;

    // printf("DEBUG: Process (%d,%d): Domain [%d:%d, %d:%d], h=%f\n", cart2d->coords[0], cart2d->coords[1],cart2d->local_start[0], cart2d->local_end[0],cart2d->local_start[1], cart2d->local_end[1], h);

    for (i = 1; i <= cart2d->local_size[0]; i++)
    {
        for (j = 1; j <= cart2d->local_size[1]; j++)
        {
            int global_i = cart2d->local_start[0] + i - 1;
            int global_j = cart2d->local_start[1] + j - 1;

            x = global_j * h;
            y = global_i * h;

            exact_sol = analytical_solution(x, y);
            error = fabs(a[i][j] - exact_sol);

            if (error > max_error)
            {
                max_error = error;
            }

            if ((i == 1 && j == 1) || (i == cart2d->local_size[0] && j == cart2d->local_size[1]))
            {
                // printf("DEBUG: Point (%d,%d): global_coords=(%d,%d), x=%f, y=%f, value=%f, exact=%f, error=%f\n",i, j, global_i, global_j, x, y, a[i][j], exact_sol, error);
            }
        }
    }

    return max_error;
}

// Free resources used by Cart2D
void free_cart2d(Cart2D *cart2d)
{
    // Free the RMA window
    MPI_Win_free(&(cart2d->grid_win));

    // Free the column datatype
    MPI_Type_free(&(cart2d->column_type));

    // Free the cartesian communicator
    MPI_Comm_free(&(cart2d->cart_comm));
}
