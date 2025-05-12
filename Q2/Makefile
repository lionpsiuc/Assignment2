CC = mpicc
CFLAGS = -Wall -g -std=c99
EXEC = matmul

all: $(EXEC)

$(EXEC): matmul.c
	$(CC) $(CFLAGS) -o $(EXEC) matmul.c

clean:
	rm -f $(EXEC) *.o
