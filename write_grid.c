#include <stdio.h>
#include <stdlib.h>
#include "write_grid.h"
#include "poisson1d.h"

/*
 * Write grid to file in mesh/grid format
 * Format: Each row corresponds to a y-coordinate (in descending order)
 * Each column corresponds to an x-coordinate (in ascending order)
 * Grid values are written in a matrix format
 */
 void write_grid(char *fname, double x[][maxn], int nx, int ny)
 {
     FILE *fp;
     int i, j;
     double h = 1.0 / (nx + 1);
     
     fp = fopen(fname, "w");
     if (!fp) {
         fprintf(stderr, "Error: can't open file %s\n", fname);
         exit(4);
     }
     
     // Write header
     fprintf(fp, "# Grid size: %d x %d\n", nx+2, ny+2);
     fprintf(fp, "# Format: x y u(x,y)\n");
     
     // Write grid points in mesh/grid format
     for (j = ny+1; j >= 0; j--) {
         double y = j * h;
         for (i = 0; i <= nx+1; i++) {
             double xcoord = i * h;
             fprintf(fp, "%lf %lf %lf\n", xcoord, y, x[i][j]);
         }
         // Add a blank line between rows to separate in gnuplot
         fprintf(fp, "\n");
     }
     
     fclose(fp);
     printf("Grid written to file: %s\n", fname);
 }