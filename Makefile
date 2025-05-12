# Compiler and flags
CC = mpicc
CFLAGS = -Wall -O3
LDFLAGS = -lm

# Executable names
EXEC1 = poisson_mp
EXEC2 = poisson_rma_fence
EXEC3 = poisson_rma_pscw

# Source files
COMMON_SRC = jacobi2d.c write_grid.c
HEADERS = poisson1d.h jacobi2d.h write_grid.h

# Object files
COMMON_OBJ = $(COMMON_SRC:.c=.o)

# Default target
all: $(EXEC1) $(EXEC2) $(EXEC3)

# Regular Message Passing version
$(EXEC1): main_q4.o $(COMMON_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# RMA with fence synchronization version
$(EXEC2): main_rma_fence.o $(COMMON_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# RMA with general active target synchronization version
$(EXEC3): main_rma_pscw.o $(COMMON_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up compiled files
clean:
	rm -f *.o $(EXEC1) $(EXEC2) $(EXEC3)

# Print help information
help:
	@echo "Makefile for MPI Programming Exercise 2 - Poisson Solver with RMA"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build all executables (default)"
	@echo "  $(EXEC1)    - Build message passing version"
	@echo "  $(EXEC2)    - Build RMA fence version"
	@echo "  $(EXEC3)    - Build RMA PSCW version"
	@echo "  clean       - Remove all compiled files"
	@echo "  help        - Display this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make                  - Build all versions"
	@echo "  make $(EXEC1)         - Build only message passing version"
	@echo "  make clean            - Clean up all compiled files"
	@echo ""
	@echo "Running examples:"
	@echo "  mpirun -np 4 ./$(EXEC1) 63           - Run MP version with 63x63 grid on 4 processes"
	@echo "  mpirun -np 4 ./$(EXEC2) 63           - Run RMA fence version with 63x63 grid on 4 processes"
	@echo "  mpirun -np 4 ./$(EXEC3) 63           - Run RMA PSCW version with 63x63 grid on 4 processes"
	@echo ""

.PHONY: all clean help