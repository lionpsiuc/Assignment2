CC = mpicc
CFLAGS = -Wall -g

all: matrix_vector

matrix_vector: matrix_vector_multiply.c
	$(CC) $(CFLAGS) -o matrix_vector matrix_vector_multiply.c

clean:
	rm -f matrix_vector    